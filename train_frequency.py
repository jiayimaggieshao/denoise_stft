#!/usr/bin/env python3
"""Train CardioSpecNet on on-the-fly, reference-conditioned synthetic mixtures."""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.frequency_data import MixingConfig, SyntheticFrequencyDataset
from src.frequency_loss import FrequencyDenoiseLoss, FrequencyLossConfig
from src.frequency_metrics import (
    MetricAccumulator,
    log_spectral_distance,
    pearson_correlation,
    si_sdr_db,
    snr_db,
)
from src.frequency_model import CardioSpecNet, FrequencyModelConfig, STFTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--step", type=str, default="0.1s", choices=("0.01s", "0.1s", "1s"))
    parser.add_argument("--source_stride", type=int, default=10,
                        help="Take every Nth stored overlapping source window (10 means effective 1 s stride for step_0.1s).")
    parser.add_argument("--samples_per_epoch", type=int, default=20_000)
    parser.add_argument("--val_samples", type=int, default=2_000)
    parser.add_argument("--max_clean_windows", type=int, default=None)
    parser.add_argument("--max_noise_windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Keep at 0 unless RAM is ample; each worker owns its window pools.")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--base_channels", type=int, default=12)
    parser.add_argument("--grid_blocks", type=int, default=2)
    parser.add_argument("--snr_min_db", type=float, default=-10.0)
    parser.add_argument("--snr_max_db", type=float, default=20.0)
    parser.add_argument("--reference_dropout", type=float, default=0.20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/frequency"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="etextile-frequency-denoise")
    parser.add_argument("--smoke", action="store_true",
                        help="Small deterministic CPU run that validates the full train/eval/checkpoint path.")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def validate(
    model: CardioSpecNet,
    criterion: FrequencyDenoiseLoss,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["chest"], batch["reference"], batch["ref_available"])
        loss, terms = criterion(output, batch["clean"])
        input_sisdr = si_sdr_db(batch["chest"], batch["clean"])
        output_sisdr = si_sdr_db(output, batch["clean"])
        input_snr = snr_db(batch["chest"], batch["clean"])
        output_snr = snr_db(output, batch["clean"])
        accumulator.update(
            loss=loss,
            input_si_sdr_db=input_sisdr,
            output_si_sdr_db=output_sisdr,
            si_sdr_improvement_db=output_sisdr - input_sisdr,
            input_snr_db=input_snr,
            output_snr_db=output_snr,
            snr_improvement_db=output_snr - input_snr,
            correlation=pearson_correlation(output, batch["clean"]),
            log_spectral_distance_db=log_spectral_distance(output, batch["clean"]),
            **{f"loss_{key}": value for key, value in terms.items() if key != "loss"},
        )
    return accumulator.summary()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.samples_per_epoch = min(args.samples_per_epoch, 256)
        args.val_samples = min(args.val_samples, 128)
        args.batch_size = min(args.batch_size, 8)
        args.base_channels = min(args.base_channels, 8)
        args.grid_blocks = min(args.grid_blocks, 1)
        args.max_clean_windows = args.max_clean_windows or 256
        args.max_noise_windows = args.max_noise_windows or 192
        args.output_dir = args.output_dir / "smoke"

    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mixing_config = MixingConfig(
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        reference_dropout_probability=args.reference_dropout,
    )
    clean_train = args.data_root / "clean" / f"step_{args.step}" / "train"
    noise_train = args.data_root / "noise" / f"step_{args.step}" / "train"
    clean_val = args.data_root / "clean" / f"step_{args.step}" / "val"
    noise_val = args.data_root / "noise" / f"step_{args.step}" / "val"

    print(f"Loading overlap-thinned pools from {clean_train} and {noise_train}")
    train_dataset = SyntheticFrequencyDataset(
        clean_train,
        noise_train,
        samples_per_epoch=args.samples_per_epoch,
        source_stride=args.source_stride,
        max_clean_windows=args.max_clean_windows,
        max_noise_windows=args.max_noise_windows,
        seed=args.seed,
        config=mixing_config,
    )
    # Validation has no identity cases and no reference dropout so methods are
    # compared at controlled SNR with a usable exterior channel.
    val_config = replace(
        mixing_config,
        identity_probability=0.0,
        reference_dropout_probability=0.0,
    )
    val_dataset = SyntheticFrequencyDataset(
        clean_val,
        noise_val,
        samples_per_epoch=args.val_samples,
        source_stride=args.source_stride,
        max_clean_windows=args.max_clean_windows,
        max_noise_windows=args.max_noise_windows,
        seed=args.seed + 10_000,
        config=val_config,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    stft_config = STFTConfig()
    model_config = FrequencyModelConfig(
        base_channels=args.base_channels,
        grid_blocks=args.grid_blocks,
    )
    model = CardioSpecNet(stft_config, model_config).to(device)
    loss_config = FrequencyLossConfig()
    criterion = FrequencyDenoiseLoss(loss_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    start_epoch = 0
    best_score = -float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_score", best_score))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device={device}; parameters={parameter_count:,}; train_pool={len(train_dataset.clean_pool):,} clean / "
          f"{len(train_dataset.noise_pool):,} noise")

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            config={**vars(args), "parameters": parameter_count},
        )

    history_path = args.output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        start_time = perf_counter()
        train_dataset.set_epoch(epoch)
        model.train()
        train_metrics = MetricAccumulator()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(batch["chest"], batch["reference"], batch["ref_available"])
                loss, terms = criterion(output, batch["clean"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_metrics.update(loss=loss, **{key: value for key, value in terms.items() if key != "loss"})
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        validation = validate(model, criterion, val_loader, device)
        score = 0.5 * (
            validation["si_sdr_improvement_db_mean"]
            + validation["snr_improvement_db_mean"]
        )
        scheduler.step(score)
        elapsed = perf_counter() - start_time
        record = {
            "epoch": epoch,
            "elapsed_seconds": elapsed,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics.summary(),
            "validation": validation,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        checkpoint = {
            "epoch": epoch,
            "best_score": max(best_score, score),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model.config_dict,
            "mixing_config": mixing_config.to_dict(),
            "loss_config": loss_config.to_dict(),
            "training_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "parameter_count": parameter_count,
            "validation": validation,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(checkpoint, args.output_dir / "best.pt")

        print(
            f"epoch={epoch + 1:03d} loss={record['train']['loss_mean']:.4f} "
            f"val_SI-SDRi={validation['si_sdr_improvement_db_mean']:.3f} dB "
            f"val_SNRi={validation['snr_improvement_db_mean']:.3f} dB "
            f"score={score:.3f} corr={validation['correlation_mean']:.3f} time={elapsed:.1f}s"
        )
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train/{key}": value for key, value in record["train"].items()},
                **{f"val/{key}": value for key, value in validation.items()},
            })

    if wandb_run is not None:
        wandb_run.finish()
    print(f"Best checkpoint: {args.output_dir / 'best.pt'} (balanced improvement score={best_score:.3f} dB)")


if __name__ == "__main__":
    # Avoid excessive thread oversubscription on shared CPU hosts.
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    main()
