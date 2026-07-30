#!/usr/bin/env python3
"""Convert PCM txt (CSV format) to WAV with optional RMS normalization."""

# python data/scripts/pcm_txt_to_wav.py
# ============ Run configuration (used when executing the script directly) ============
# Input txt: data/dataset/raw/ (DEFAULT_RAW_DIR in pcm_paths.py)

# File stem(s) under raw/, e.g. "heart_bw1" or ["heart_bw1", "noise2"]
SELECT_FILES = "heart_bw3"

# Enable RMS normalization
ENABLE_RMS_NORMALIZATION = False

# Target level in dBFS (only when ENABLE_RMS_NORMALIZATION = True)
TARGET_RMS_DB = -20.0

# Use preprocessed txt from raw/{stem}_preprocessed.txt
USE_PREPROCESSED = False
# ====================================================================================

import argparse
import csv
import math
import sys
import wave
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pcm_paths import (
    DEFAULT_DATASET_DIR as DATASET_DIR,
    DEFAULT_RAW_DIR as RAW_DIR,
    resolve_wav_jobs,
    wav_output_for_input,
)

INT16_MAX = 32767


def infer_sample_rate(input_path: Path) -> int:
    """Infer sample rate (Hz) from consecutive mono_us differences."""
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_names = [h.strip() for h in header]
        mono_idx = col_names.index("mono_us")

        prev_ts = None
        diffs = []
        for i, row in enumerate(reader):
            ts = int(row[mono_idx].strip())
            if prev_ts is not None:
                diffs.append(ts - prev_ts)
            prev_ts = ts
            if i >= 999:
                break

    if not diffs:
        raise ValueError("Cannot infer sample rate from mono_us; specify --sample-rate")

    median_diff_us = sorted(diffs)[len(diffs) // 2]
    return int(round(1_000_000 / median_diff_us))


def rms_normalize(samples: list[int], target_rms_db: float) -> tuple[list[int], float, float]:
    """Normalize sample RMS to target dBFS and clip to int16 range."""
    floats = [float(s) for s in samples]
    rms = math.sqrt(sum(x * x for x in floats) / len(floats))
    if rms == 0:
        raise ValueError("Sample RMS is 0; cannot apply normalization")

    target_rms = (10 ** (target_rms_db / 20.0)) * INT16_MAX
    gain = target_rms / rms

    normalized: list[int] = []
    for x in floats:
        val = int(round(x * gain))
        normalized.append(max(-32768, min(INT16_MAX, val)))

    output_rms = math.sqrt(sum(x * x for x in normalized) / len(normalized))
    return normalized, rms, output_rms


def pcm_txt_to_wav(
    input_path: Path,
    output_path: Path,
    sample_rate: int | None = None,
    *,
    enable_rms_normalization: bool = True,
    target_rms_db: float = -20.0,
) -> None:
    if sample_rate is None:
        sample_rate = infer_sample_rate(input_path)

    samples: list[int] = []
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_names = [h.strip() for h in header]
        x_idx = col_names.index("x")

        for row in reader:
            samples.append(int(row[x_idx].strip()))

    if not samples:
        raise ValueError(f"No samples read from {input_path}")

    input_rms = math.sqrt(sum(x * x for x in samples) / len(samples))
    if enable_rms_normalization:
        samples, input_rms, output_rms = rms_normalize(samples, target_rms_db)
    else:
        output_rms = input_rms

    frames = b"".join(
        sample.to_bytes(2, byteorder="little", signed=True) for sample in samples
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)

    duration_s = len(samples) / sample_rate
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Samples: {len(samples)}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration_s:.2f} s")
    if enable_rms_normalization:
        print(f"RMS normalization: ON, target {target_rms_db:.1f} dBFS")
        print(f"Input RMS: {input_rms:.2f} -> Output RMS: {output_rms:.2f}")
    else:
        print("RMS normalization: OFF (raw PCM amplitude)")
        print(f"RMS: {input_rms:.2f}")


def resolve_rms_settings(args: argparse.Namespace) -> tuple[bool, float]:
    enable_rms = (
        args.enable_rms_normalization
        if args.enable_rms_normalization is not None
        else ENABLE_RMS_NORMALIZATION
    )
    return enable_rms, args.target_rms_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PCM txt to WAV (optional RMS normalization)")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Input txt path (default: SELECT_FILES at top of script)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output wav path (default: dataset/audio/{stem}.wav)",
    )
    parser.add_argument(
        "-r",
        "--sample-rate",
        type=int,
        default=None,
        help="Sample rate in Hz (default: infer from mono_us)",
    )
    parser.add_argument(
        "--target-rms-db",
        type=float,
        default=TARGET_RMS_DB,
        help=f"RMS normalization target in dBFS (default: {TARGET_RMS_DB})",
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help=f"Use preprocessed txt (default: USE_PREPROCESSED = {USE_PREPROCESSED})",
    )
    parser.add_argument(
        "--rms-normalization",
        dest="enable_rms_normalization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable RMS normalization "
            f"(default: ENABLE_RMS_NORMALIZATION = {ENABLE_RMS_NORMALIZATION})"
        ),
    )
    args = parser.parse_args()
    enable_rms, target_rms_db = resolve_rms_settings(args)
    use_preprocessed = args.preprocessed or USE_PREPROCESSED

    if args.input is not None:
        input_path = args.input.resolve()
        output_path = (
            args.output.resolve()
            if args.output is not None
            else wav_output_for_input(input_path, DATASET_DIR)
        )
        pcm_txt_to_wav(
            input_path,
            output_path,
            sample_rate=args.sample_rate,
            enable_rms_normalization=enable_rms,
            target_rms_db=target_rms_db,
        )
        return

    jobs = resolve_wav_jobs(SELECT_FILES, DATASET_DIR, use_preprocessed=use_preprocessed)

    print("Using script configuration:")
    print(f"  RAW_DIR                  = {RAW_DIR}")
    print(f"  SELECT_FILES             = {SELECT_FILES}")
    print(f"  USE_PREPROCESSED         = {use_preprocessed}")
    print(f"  ENABLE_RMS_NORMALIZATION = {enable_rms}")
    if enable_rms:
        print(f"  TARGET_RMS_DB            = {target_rms_db}")

    processed = 0
    for _stem, input_path, default_output in jobs:
        if not input_path.is_file():
            print(f"[SKIP] File not found: {input_path}", file=sys.stderr)
            continue
        output_path = (
            args.output.resolve()
            if args.output is not None and len(jobs) == 1
            else default_output
        )
        print()
        pcm_txt_to_wav(
            input_path,
            output_path,
            sample_rate=args.sample_rate,
            enable_rms_normalization=enable_rms,
            target_rms_db=target_rms_db,
        )
        processed += 1

    if processed == 0:
        print("No files processed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
