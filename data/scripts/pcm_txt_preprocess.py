#!/usr/bin/env python3
"""Preprocess PCM txt: remove DC + bandpass filter, write txt to raw/."""

# python data/scripts/pcm_txt_preprocess.py
# ============ Run configuration (used when executing the script directly) ============
# Input txt: data/dataset/raw/ (DEFAULT_RAW_DIR in pcm_paths.py)

# File stem(s) under raw/, e.g. "heart_bw1" or ["heart_bw1", "noise2"]
SELECT_FILES = "heart_aw1"

SAMPLE_RATE = 4000

# Remove DC: x = x - median(x)
ENABLE_DC_REMOVAL = True

# Bandpass filter
ENABLE_BANDPASS = True
BANDPASS_LOW_HZ = 20.0
BANDPASS_HIGH_HZ = 500.0
BANDPASS_ORDER = 4
# ====================================================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pcm_paths import (
    DEFAULT_DATASET_DIR as DATASET_DIR,
    DEFAULT_RAW_DIR as RAW_DIR,
    preprocess_output_for_input,
    resolve_preprocess_jobs,
)

EXPECTED_HEADER = ("idx", "mono_us", "wall_epoch_us", "x")


def load_pcm_txt(input_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_names = tuple(h.strip() for h in header)
        if col_names != EXPECTED_HEADER:
            raise ValueError(f"Header mismatch: expected {EXPECTED_HEADER}, got {col_names}")

        idx_list: list[int] = []
        mono_list: list[int] = []
        wall_list: list[int] = []
        x_list: list[int] = []

        for row_number, row in enumerate(reader, start=2):
            if len(row) < 4:
                raise ValueError(f"Row {row_number} has too few columns")
            try:
                idx_list.append(int(row[0].strip()))
                mono_list.append(int(row[1].strip()))
                wall_list.append(int(row[2].strip()))
                x_list.append(int(row[3].strip()))
            except ValueError as exc:
                raise ValueError(f"Failed to parse row {row_number}: {exc}") from exc

    if not x_list:
        raise ValueError(f"No samples read from {input_path}")

    return (
        np.asarray(idx_list, dtype=np.int64),
        np.asarray(mono_list, dtype=np.int64),
        np.asarray(wall_list, dtype=np.int64),
        np.asarray(x_list, dtype=np.float32),
    )


def validate_bandpass_params(
    low_hz: float,
    high_hz: float,
    order: int,
    sample_rate: int,
) -> None:
    if order <= 0:
        raise ValueError(f"BANDPASS_ORDER must be > 0, got {order}")
    if low_hz <= 0:
        raise ValueError(f"BANDPASS_LOW_HZ must be > 0, got {low_hz}")
    if high_hz <= low_hz:
        raise ValueError(
            f"BANDPASS_HIGH_HZ must be > BANDPASS_LOW_HZ, got {low_hz} .. {high_hz}"
        )
    nyquist = sample_rate / 2
    if high_hz >= nyquist:
        raise ValueError(
            f"BANDPASS_HIGH_HZ must be < Nyquist ({nyquist} Hz), got {high_hz}"
        )


def design_bandpass_sos(
    low_hz: float,
    high_hz: float,
    order: int,
    sample_rate: int,
) -> np.ndarray:
    validate_bandpass_params(low_hz, high_hz, order, sample_rate)
    return butter(
        order,
        [low_hz, high_hz],
        btype="band",
        fs=sample_rate,
        output="sos",
    )


def preprocess_signal(
    x: np.ndarray,
    *,
    enable_dc_removal: bool,
    enable_bandpass: bool,
    bandpass_sos: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    processed = x.astype(np.float64, copy=True)
    dc_offset = 0.0

    if enable_dc_removal:
        dc_offset = float(np.median(processed))
        processed -= dc_offset

    if enable_bandpass:
        if bandpass_sos is None:
            raise ValueError("enable_bandpass=True but bandpass_sos was not provided")
        processed = sosfiltfilt(bandpass_sos, processed)

    return processed.astype(np.float32), dc_offset


def write_pcm_txt(
    output_path: Path,
    idx: np.ndarray,
    mono_us: np.ndarray,
    wall_epoch_us: np.ndarray,
    x: np.ndarray,
) -> None:
    x_int = np.clip(np.round(x), -32768, 32767).astype(np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        f.write("idx, mono_us, wall_epoch_us, x\n")
        for i in range(len(idx)):
            f.write(f"{idx[i]}, {mono_us[i]}, {wall_epoch_us[i]}, {x_int[i]}\n")


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    sample_rate: int,
    enable_dc_removal: bool,
    enable_bandpass: bool,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    bandpass_order: int,
) -> dict[str, str | int | float]:
    idx, mono_us, wall_epoch_us, x = load_pcm_txt(input_path)

    bandpass_sos = None
    if enable_bandpass:
        bandpass_sos = design_bandpass_sos(
            bandpass_low_hz,
            bandpass_high_hz,
            bandpass_order,
            sample_rate,
        )

    x_processed, dc_offset = preprocess_signal(
        x,
        enable_dc_removal=enable_dc_removal,
        enable_bandpass=enable_bandpass,
        bandpass_sos=bandpass_sos,
    )

    write_pcm_txt(output_path, idx, mono_us, wall_epoch_us, x_processed)

    duration_s = len(x_processed) / sample_rate
    return {
        "input": str(input_path),
        "output": str(output_path),
        "num_samples": len(x_processed),
        "duration_s": duration_s,
        "dc_offset": dc_offset,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess PCM txt: remove DC + Butterworth bandpass (filtfilt)"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input txt path(s) (default: SELECT_FILES at top of script)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help=f"Sample rate in Hz (default: {SAMPLE_RATE})",
    )
    parser.add_argument(
        "--no-dc-removal",
        action="store_true",
        help="Skip DC removal (default: use ENABLE_DC_REMOVAL)",
    )
    parser.add_argument(
        "--no-bandpass",
        action="store_true",
        help="Skip bandpass (default: use ENABLE_BANDPASS)",
    )
    parser.add_argument(
        "--bandpass-low-hz",
        type=float,
        default=None,
        help=f"Bandpass low cutoff in Hz (default: {BANDPASS_LOW_HZ})",
    )
    parser.add_argument(
        "--bandpass-high-hz",
        type=float,
        default=None,
        help=f"Bandpass high cutoff in Hz (default: {BANDPASS_HIGH_HZ})",
    )
    parser.add_argument(
        "--bandpass-order",
        type=int,
        default=None,
        help=f"Butterworth filter order (default: {BANDPASS_ORDER})",
    )
    args = parser.parse_args()

    sample_rate = args.sample_rate if args.sample_rate is not None else SAMPLE_RATE
    enable_dc_removal = ENABLE_DC_REMOVAL and not args.no_dc_removal
    enable_bandpass = ENABLE_BANDPASS and not args.no_bandpass
    bandpass_low_hz = (
        args.bandpass_low_hz if args.bandpass_low_hz is not None else BANDPASS_LOW_HZ
    )
    bandpass_high_hz = (
        args.bandpass_high_hz if args.bandpass_high_hz is not None else BANDPASS_HIGH_HZ
    )
    bandpass_order = (
        args.bandpass_order if args.bandpass_order is not None else BANDPASS_ORDER
    )

    if args.inputs:
        jobs: list[tuple[str, Path, Path]] = []
        for input_path in args.inputs:
            resolved = input_path.resolve()
            out_path = preprocess_output_for_input(resolved, DATASET_DIR)
            jobs.append((resolved.stem, resolved, out_path))
    else:
        jobs = resolve_preprocess_jobs(SELECT_FILES, DATASET_DIR)
        print("Using script configuration:")
        print(f"  RAW_DIR           = {RAW_DIR}")
        print(f"  SELECT_FILES      = {SELECT_FILES}")
        print(f"  SAMPLE_RATE       = {sample_rate}")
        print(f"  ENABLE_DC_REMOVAL = {enable_dc_removal}")
        print(f"  ENABLE_BANDPASS   = {enable_bandpass}")
        if enable_bandpass:
            print(
                f"  BANDPASS          = {bandpass_low_hz}-{bandpass_high_hz} Hz, "
                f"order={bandpass_order}, filtfilt"
            )

    summaries: list[dict[str, str | int | float]] = []
    for _stem, input_path, output_path in jobs:
        if not input_path.is_file():
            print(f"[SKIP] File not found: {input_path}", file=sys.stderr)
            continue

        print(f"\nProcessing: {input_path}")
        summary = process_file(
            input_path,
            output_path,
            sample_rate=sample_rate,
            enable_dc_removal=enable_dc_removal,
            enable_bandpass=enable_bandpass,
            bandpass_low_hz=bandpass_low_hz,
            bandpass_high_hz=bandpass_high_hz,
            bandpass_order=bandpass_order,
        )
        summaries.append(summary)
        print(f"  Samples: {summary['num_samples']}")
        print(f"  Duration: {summary['duration_s']:.2f} s")
        if enable_dc_removal:
            print(f"  DC removal: median = {summary['dc_offset']:.4f}")
        if enable_bandpass:
            print(
                f"  Bandpass: {bandpass_low_hz}-{bandpass_high_hz} Hz, "
                f"Butterworth order={bandpass_order}, filtfilt"
            )
        print(f"  Saved to: {summary['output']}")

    if not summaries:
        print("No files processed", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 72}")
    print(f"Processed {len(summaries)} file(s)")


if __name__ == "__main__":
    main()
