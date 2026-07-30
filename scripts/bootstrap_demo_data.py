#!/usr/bin/env python3
"""Create minimal synthetic npz archives so smoke training can run without real recordings."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_demo_npz(path: Path, count: int, seed: int, amplitude: float) -> None:
    rng = np.random.default_rng(seed)
    windows = (rng.normal(0.0, amplitude, size=(count, 8_000))).astype(np.int16)
    start_idx = np.arange(0, count * 400, 400, dtype=np.int64)
    segment_id = np.zeros(count, dtype=np.int32)
    wall_time = (np.arange(count, dtype=np.int64) * 100_000) + 1_700_000_000_000_000
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=windows,
        start_idx=start_idx,
        segment_id=segment_id,
        start_wall_epoch_us=wall_time,
    )
    print(f"Wrote {path} ({count} windows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--step", default="0.1s")
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()

    step = args.step if args.step.startswith("step_") else f"step_{args.step}"
    specs = [
        (args.data_root / "clean" / step / "train" / "demo_clean_train.npz", 101, 500.0),
        (args.data_root / "clean" / step / "val" / "demo_clean_val.npz", 102, 500.0),
        (args.data_root / "noise" / step / "train" / "demo_noise_train.npz", 103, 2000.0),
        (args.data_root / "noise" / step / "val" / "demo_noise_val.npz", 104, 2000.0),
    ]
    for path, seed, amplitude in specs:
        write_demo_npz(path, args.count, seed, amplitude)
    print("Demo npz ready. Run: python train_frequency.py --smoke --device cpu")


if __name__ == "__main__":
    main()
