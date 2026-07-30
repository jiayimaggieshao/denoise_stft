#!/usr/bin/env python3
"""Denoise real walking PCG windows with timestamp-aligned exterior reference audio."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from src.frequency_baselines import (
    bandpass_only,
    complex_transfer_cancellation,
    hybrid_frequency_fusion,
    reference_spectral_subtraction,
)
from src.frequency_model import build_frequency_model
from src.window_alignment import (
    load_window_archive,
    nearest_timestamp_alignment,
    overlap_add_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Chest-facing walking-window NPZ")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Exterior-microphone NPZ. Omit to auto-find noise2/3/5/6.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--segment", type=int, default=None, help="Segment ID; default is the first segment")
    parser.add_argument("--start", type=float, default=0.0, help="Seconds from selected segment start")
    parser.add_argument("--end", type=float, default=20.0)
    parser.add_argument("--sample_rate", type=int, default=4_000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_alignment_ms", type=float, default=100.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--primary_method",
        choices=(
            "hybrid_fusion",
            "cardiospecnet",
            "complex_transfer_cancellation",
            "reference_subtraction",
            "bandpass",
        ),
        default="hybrid_fusion",
        help="Method written to denoised.wav. The default monitoring profile fuses neural and adaptive outputs.",
    )
    parser.add_argument(
        "--fusion_neural_weight",
        type=float,
        default=0.55,
        help="CardioSpecNet weight in hybrid_fusion (0.55 monitoring; 0.20 raw-fidelity calibration).",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--export_baselines",
        action="store_true",
        help="Also export bandpass, reference-subtraction, and complex-transfer WAVs.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def to_float(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        return x.astype(np.float32) / float(max(abs(info.min), info.max))
    return x.astype(np.float32)


def auto_reference(chest_path: Path) -> Path | None:
    match = re.search(r"heart_w([2356])_windows", chest_path.stem)
    if not match:
        return None
    subject = match.group(1)
    repo_root = Path(__file__).resolve().parents[1]
    step_dir = chest_path.parent.name if chest_path.parent.name.startswith("step_") else "step_0.1s"
    for split in ("train", "val"):
        candidate = repo_root / "data" / "noise" / step_dir / split / f"noise{subject}_windows.npz"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    if args.end <= args.start:
        raise ValueError("--end must be greater than --start")
    device = choose_device(args.device)
    chest_archive = load_window_archive(args.input)
    reference_path = args.reference or auto_reference(args.input)
    reference_archive = load_window_archive(reference_path) if reference_path else None

    segment = int(np.unique(chest_archive.segment_id)[0]) if args.segment is None else args.segment
    segment_mask = chest_archive.segment_id == segment
    if not np.any(segment_mask):
        raise ValueError(f"segment {segment} not present; available={np.unique(chest_archive.segment_id).tolist()}")
    segment_min_start = int(chest_archive.start_idx[segment_mask].min())
    start_sample = segment_min_start + int(round(args.start * args.sample_rate))
    end_sample = segment_min_start + int(round(args.end * args.sample_rate))
    # Include any window that overlaps the requested interval.
    window_length = chest_archive.x.shape[1]
    select = segment_mask & (chest_archive.start_idx < end_sample) & (
        chest_archive.start_idx + window_length > start_sample
    )
    selected_indices = np.flatnonzero(select)
    if len(selected_indices) == 0:
        raise ValueError("No windows overlap the requested interval")

    chest_windows = to_float(chest_archive.x[selected_indices])
    starts = chest_archive.start_idx[selected_indices]
    times = chest_archive.wall_time_us[selected_indices]

    aligned_reference = np.zeros_like(chest_windows)
    available = np.zeros(len(chest_windows), dtype=np.float32)
    alignment_deltas_ms: list[float] = []
    if reference_archive is not None:
        reference_indices, matched = nearest_timestamp_alignment(
            times,
            reference_archive.wall_time_us,
            max_delta_us=int(round(args.max_alignment_ms * 1_000.0)),
        )
        aligned_reference[matched] = to_float(reference_archive.x[reference_indices[matched]])
        available[matched] = 1.0
        alignment_deltas_ms = (
            np.abs(reference_archive.wall_time_us[reference_indices[matched]] - times[matched]) / 1_000.0
        ).tolist()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_frequency_model(checkpoint.get("model_config")).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if not 0.0 <= args.fusion_neural_weight <= 1.0:
        raise ValueError("--fusion_neural_weight must be in [0, 1]")

    primary_batches: list[np.ndarray] = []
    neural_batches: list[np.ndarray] = []
    baseline_batches: dict[str, list[np.ndarray]] = {
        "bandpass": [],
        "reference_subtraction": [],
        "complex_transfer_cancellation": [],
    }
    with torch.inference_mode():
        for begin in range(0, len(chest_windows), args.batch_size):
            end = begin + args.batch_size
            chest_tensor = torch.from_numpy(chest_windows[begin:end]).to(device)
            reference_tensor = torch.from_numpy(aligned_reference[begin:end]).to(device)
            available_tensor = torch.from_numpy(available[begin:end]).to(device)

            neural = model(chest_tensor, reference_tensor, available_tensor)
            bandpass = bandpass_only(chest_tensor)
            subtraction = reference_spectral_subtraction(chest_tensor, reference_tensor)
            transfer = complex_transfer_cancellation(chest_tensor, reference_tensor)
            hybrid = hybrid_frequency_fusion(
                neural,
                transfer,
                neural_weight=args.fusion_neural_weight,
                ref_available=available_tensor,
            )
            estimates = {
                "hybrid_fusion": hybrid,
                "cardiospecnet": neural,
                "complex_transfer_cancellation": transfer,
                "reference_subtraction": subtraction,
                "bandpass": bandpass,
            }
            primary_batches.append(estimates[args.primary_method].cpu().numpy())
            neural_batches.append(neural.cpu().numpy())
            if args.export_baselines:
                baseline_batches["bandpass"].append(bandpass.cpu().numpy())
                baseline_batches["reference_subtraction"].append(subtraction.cpu().numpy())
                baseline_batches["complex_transfer_cancellation"].append(transfer.cpu().numpy())
    denoised_windows = np.concatenate(primary_batches, axis=0)
    neural_windows = np.concatenate(neural_batches, axis=0)

    chest_audio = overlap_add_windows(chest_windows, starts)
    denoised_audio = overlap_add_windows(denoised_windows, starts)
    neural_audio = overlap_add_windows(neural_windows, starts)
    reference_audio = overlap_add_windows(aligned_reference, starts)
    baseline_audio: dict[str, np.ndarray] = {}
    if args.export_baselines:
        baseline_audio = {
            name: overlap_add_windows(np.concatenate(batches, axis=0), starts)
            for name, batches in baseline_batches.items()
        }
    local_origin = int(starts.min())
    crop_start = max(0, start_sample - local_origin)
    crop_end = min(len(chest_audio), end_sample - local_origin)
    chest_audio = chest_audio[crop_start:crop_end]
    denoised_audio = denoised_audio[crop_start:crop_end]
    neural_audio = neural_audio[crop_start:crop_end]
    reference_audio = reference_audio[crop_start:crop_end]
    if args.export_baselines:
        baseline_audio = {
            name: audio[crop_start:crop_end]
            for name, audio in baseline_audio.items()
        }

    input_label = args.input.stem.removesuffix("_windows")
    output_dir = args.output_dir or Path("outputs") / f"{input_label}_frequency_{args.start:g}-{args.end:g}s"
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(output_dir / "original_chest.wav", chest_audio, args.sample_rate)
    sf.write(output_dir / "denoised.wav", denoised_audio, args.sample_rate)
    if args.primary_method != "cardiospecnet":
        sf.write(output_dir / "cardiospecnet.wav", neural_audio, args.sample_rate)
    sf.write(output_dir / "aligned_reference.wav", reference_audio, args.sample_rate)
    if args.export_baselines:
        for name, audio in baseline_audio.items():
            sf.write(output_dir / f"{name}.wav", audio, args.sample_rate)

    time = np.arange(len(chest_audio)) / args.sample_rate + args.start
    n_rows = 4 if args.export_baselines else 3
    figure, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows), sharex=True)
    axes[0].plot(time, chest_audio, linewidth=0.6)
    axes[0].set_title("Chest microphone (walking)")
    axes[1].plot(time, denoised_audio, linewidth=0.6)
    axes[1].set_title(f"Primary output: {args.primary_method}")
    spectrogram_axis = axes[2]
    waveform_axes = list(axes[:2])
    if args.export_baselines:
        axes[2].plot(time, baseline_audio["complex_transfer_cancellation"], linewidth=0.6)
        axes[2].set_title("Complex transfer cancellation baseline")
        spectrogram_axis = axes[3]
        waveform_axes.append(axes[2])
    spectrogram_axis.specgram(denoised_audio, NFFT=512, Fs=args.sample_rate, noverlap=448)
    spectrogram_axis.set_ylim(0, 1_000)
    spectrogram_axis.set_title(f"{args.primary_method} output spectrogram")
    spectrogram_axis.set_xlabel("Time (s)")
    for axis in waveform_axes:
        axis.set_ylabel("Amplitude")
    spectrogram_axis.set_ylabel("Frequency (Hz)")
    figure.tight_layout()
    figure.savefig(output_dir / "comparison.png", dpi=160)
    plt.close(figure)

    metadata = {
        "input": str(args.input),
        "reference": str(reference_path) if reference_path else None,
        "checkpoint": str(args.checkpoint),
        "primary_method": args.primary_method,
        "fusion_neural_weight": args.fusion_neural_weight,
        "missing_reference_fallback": "cardiospecnet",
        "segment": segment,
        "requested_seconds": [args.start, args.end],
        "windows_processed": len(chest_windows),
        "reference_match_fraction": float(available.mean()),
        "alignment_delta_ms_median": float(np.median(alignment_deltas_ms)) if alignment_deltas_ms else None,
        "alignment_delta_ms_p95": float(np.percentile(alignment_deltas_ms, 95)) if alignment_deltas_ms else None,
        "exported_baselines": sorted(baseline_audio) if args.export_baselines else [],
        "note": "No ground-truth clean target exists for walking recordings; output quality must be reviewed clinically and with proxy metrics.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
