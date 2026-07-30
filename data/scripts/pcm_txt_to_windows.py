#!/usr/bin/env python3
"""Split PCM txt at idx gaps, window (2s window / configurable step), write npz under windows/step_*/."""

# python data/scripts/pcm_txt_to_windows.py
# ============ Run configuration (used when executing the script directly) ============
# Input txt: data/dataset/raw/ (DEFAULT_RAW_DIR in pcm_paths.py)

# File stem(s) under raw/, e.g. "heart_bw1" or ["heart_bw1", "noise2"]
SELECT_FILES = "heart_w6"

# Drop this many seconds from the start before windowing
SKIP_HEAD_SECONDS = 5.0

# Sliding-window step in seconds: 0.01, 0.1, or 1.0
# Output: data/dataset/windows/step_0.01s | step_0.1s | step_1s
WINDOW_STEP_SECONDS = 0.1

# Use preprocessed txt from raw/{stem}_preprocessed.txt
USE_PREPROCESSED = False
# ====================================================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pcm_paths import (
    DEFAULT_DATASET_DIR as DATASET_DIR,
    DEFAULT_RAW_DIR as RAW_DIR,
    file_stem_from_path,
    resolve_windows_jobs,
    step_windows_subdir,
    windows_output_for_input,
)

SAMPLE_RATE = 4000
WINDOW_SIZE = 8000   # 2 s @ 4 kHz


def window_step_samples(step_seconds: float) -> int:
    step = int(round(step_seconds * SAMPLE_RATE))
    if step <= 0:
        raise ValueError(f"WINDOW_STEP_SECONDS must be positive, got {step_seconds}")
    return step


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

    if not idx_list:
        raise ValueError(f"No samples read from {input_path}")

    return (
        np.asarray(idx_list, dtype=np.int64),
        np.asarray(mono_list, dtype=np.int64),
        np.asarray(wall_list, dtype=np.int64),
        np.asarray(x_list, dtype=np.int16),
    )


def skip_head(
    idx: np.ndarray,
    mono_us: np.ndarray,
    wall_epoch_us: np.ndarray,
    x: np.ndarray,
    skip_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if skip_seconds <= 0:
        return idx, mono_us, wall_epoch_us, x, 0

    skip_samples = int(round(skip_seconds * SAMPLE_RATE))
    if skip_samples <= 0:
        return idx, mono_us, wall_epoch_us, x, 0
    if skip_samples >= len(idx):
        raise ValueError(
            f"SKIP_HEAD_SECONDS={skip_seconds} requires {skip_samples} samples, "
            f"but file only has {len(idx)}"
        )

    return idx[skip_samples:], mono_us[skip_samples:], wall_epoch_us[skip_samples:], x[skip_samples:], skip_samples


def split_segment_ranges(idx: np.ndarray) -> list[tuple[int, int]]:
    if len(idx) == 0:
        return []

    gap_after = np.where(np.diff(idx) != 1)[0]
    starts = np.concatenate(([0], gap_after + 1))
    ends = np.concatenate((gap_after + 1, [len(idx)]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def verify_window_continuity(window_idx: np.ndarray) -> None:
    if len(window_idx) != WINDOW_SIZE:
        raise ValueError(f"Window must have {WINDOW_SIZE} samples, got {len(window_idx)}")
    if window_idx[-1] - window_idx[0] + 1 != WINDOW_SIZE:
        raise ValueError(
            f"Window idx not contiguous: {window_idx[0]} .. {window_idx[-1]}"
        )


def extract_windows(
    idx: np.ndarray,
    wall_epoch_us: np.ndarray,
    x: np.ndarray,
    *,
    window_step: int,
) -> dict[str, np.ndarray | int]:
    segment_ranges = split_segment_ranges(idx)

    windows_x: list[np.ndarray] = []
    start_idx_list: list[int] = []
    start_wall_list: list[int] = []
    segment_id_list: list[int] = []

    skipped_short_segments = 0
    skipped_short_samples = 0
    usable_unique_samples = 0

    for segment_id, (seg_start, seg_end) in enumerate(segment_ranges):
        seg_len = seg_end - seg_start
        if seg_len < WINDOW_SIZE:
            skipped_short_segments += 1
            skipped_short_samples += seg_len
            continue

        seg_idx = idx[seg_start:seg_end]
        seg_wall = wall_epoch_us[seg_start:seg_end]
        seg_x = x[seg_start:seg_end]

        window_in_segment = 0
        for offset in range(0, seg_len - WINDOW_SIZE + 1, window_step):
            win_start = offset
            win_end = offset + WINDOW_SIZE
            window_idx = seg_idx[win_start:win_end]

            verify_window_continuity(window_idx)

            windows_x.append(seg_x[win_start:win_end])
            start_idx_list.append(int(window_idx[0]))
            start_wall_list.append(int(seg_wall[win_start]))
            segment_id_list.append(segment_id)
            window_in_segment += 1

        if window_in_segment > 0:
            usable_unique_samples += (
                (window_in_segment - 1) * window_step + WINDOW_SIZE
            )

    num_windows = len(windows_x)
    if num_windows == 0:
        return {
            "x": np.empty((0, WINDOW_SIZE), dtype=np.int16),
            "start_idx": np.empty(0, dtype=np.int64),
            "start_wall_epoch_us": np.empty(0, dtype=np.int64),
            "segment_id": np.empty(0, dtype=np.int32),
            "num_segments": len(segment_ranges),
            "num_windows": 0,
            "usable_unique_samples": 0,
            "skipped_short_segments": skipped_short_segments,
            "skipped_short_samples": skipped_short_samples,
        }

    return {
        "x": np.stack(windows_x, axis=0),
        "start_idx": np.asarray(start_idx_list, dtype=np.int64),
        "start_wall_epoch_us": np.asarray(start_wall_list, dtype=np.int64),
        "segment_id": np.asarray(segment_id_list, dtype=np.int32),
        "num_segments": len(segment_ranges),
        "num_windows": num_windows,
        "usable_unique_samples": usable_unique_samples,
        "skipped_short_segments": skipped_short_segments,
        "skipped_short_samples": skipped_short_samples,
    }


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60.0)
    if minutes >= 1:
        return f"{seconds:.2f} s ({int(minutes)} min {secs:.2f} s)"
    return f"{seconds:.2f} s"


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    skip_head_seconds: float = SKIP_HEAD_SECONDS,
    step_seconds: float = WINDOW_STEP_SECONDS,
) -> dict[str, int | str | float]:
    window_step = window_step_samples(step_seconds)
    idx, mono_us, wall_epoch_us, x = load_pcm_txt(input_path)
    total_samples = len(idx)
    idx, _mono_us, wall_epoch_us, x, skipped_head_samples = skip_head(
        idx, mono_us, wall_epoch_us, x, skip_head_seconds
    )
    gap_count = int(np.sum(np.diff(idx) != 1))

    result = extract_windows(idx, wall_epoch_us, x, window_step=window_step)
    num_windows = int(result["num_windows"])

    usable_unique_samples = int(result["usable_unique_samples"])
    usable_duration_s = usable_unique_samples / SAMPLE_RATE

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=result["x"],
        start_idx=result["start_idx"],
        start_wall_epoch_us=result["start_wall_epoch_us"],
        segment_id=result["segment_id"],
    )

    return {
        "input": str(input_path),
        "output": str(output_path),
        "step_seconds": step_seconds,
        "window_step_samples": window_step,
        "total_samples": total_samples,
        "skipped_head_samples": skipped_head_samples,
        "skipped_head_seconds": skipped_head_samples / SAMPLE_RATE,
        "gap_count": gap_count,
        "num_segments": int(result["num_segments"]),
        "num_windows": num_windows,
        "usable_unique_samples": usable_unique_samples,
        "usable_duration_s": usable_duration_s,
        "skipped_short_segments": int(result["skipped_short_segments"]),
        "skipped_short_samples": int(result["skipped_short_samples"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split PCM txt at idx gaps, window 2s with configurable step, output npz windows"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input txt path(s) (default: SELECT_FILES at top of script)",
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help=f"Use preprocessed txt (default: USE_PREPROCESSED = {USE_PREPROCESSED})",
    )
    parser.add_argument(
        "--skip-head-seconds",
        type=float,
        default=None,
        help=f"Seconds to drop from start before windowing (default: {SKIP_HEAD_SECONDS})",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=None,
        help=(
            f"Sliding-window step in seconds (default: WINDOW_STEP_SECONDS = {WINDOW_STEP_SECONDS}). "
            "Output dir: windows/step_{step}s/"
        ),
    )
    args = parser.parse_args()
    skip_head_seconds = (
        args.skip_head_seconds if args.skip_head_seconds is not None else SKIP_HEAD_SECONDS
    )
    step_seconds = args.step_seconds if args.step_seconds is not None else WINDOW_STEP_SECONDS
    window_step = window_step_samples(step_seconds)
    use_preprocessed = args.preprocessed or USE_PREPROCESSED

    if args.inputs:
        jobs: list[tuple[str, Path, Path]] = []
        for input_path in args.inputs:
            resolved = input_path.resolve()
            out_path = windows_output_for_input(
                resolved, DATASET_DIR, step_seconds=step_seconds
            )
            jobs.append((file_stem_from_path(resolved), resolved, out_path))
    else:
        jobs = resolve_windows_jobs(
            SELECT_FILES, DATASET_DIR, use_preprocessed=use_preprocessed, step_seconds=step_seconds
        )
        print("Using script configuration:")
        print(f"  RAW_DIR             = {RAW_DIR}")
        print(f"  SELECT_FILES        = {SELECT_FILES}")
        print(f"  USE_PREPROCESSED    = {use_preprocessed}")
        print(f"  SKIP_HEAD_SECONDS   = {skip_head_seconds}")
        print(f"  WINDOW_STEP_SECONDS = {step_seconds} ({window_step} samples)")
        print(f"  OUTPUT_DIR          = {DATASET_DIR / 'windows' / step_windows_subdir(step_seconds)}")

    summaries: list[dict[str, int | str | float]] = []
    for _file_stem, input_path, output_path in jobs:
        if not input_path.is_file():
            print(f"[SKIP] File not found: {input_path}", file=sys.stderr)
            continue
        print(f"\nProcessing: {input_path}")
        summary = process_file(
            input_path,
            output_path,
            skip_head_seconds=skip_head_seconds,
            step_seconds=step_seconds,
        )
        summaries.append(summary)
        print(
            f"  Dropped head: {summary['skipped_head_samples']} samples"
            f" ({summary['skipped_head_seconds']:.2f} s)"
        )
        print(f"  Idx gaps: {summary['gap_count']}")
        print(f"  Segments: {summary['num_segments']}")
        print(
            f"  Short segments (< {WINDOW_SIZE}): {summary['skipped_short_segments']}"
            f" ({summary['skipped_short_samples']} samples total)"
        )
        print(f"  Window step: {summary['step_seconds']} s ({summary['window_step_samples']} samples)")
        print(f"  Output windows: {summary['num_windows']} ({WINDOW_SIZE} samples each)")
        print(
            f"  Usable data (overlap removed): {summary['usable_unique_samples']} samples"
            f" = {format_duration(float(summary['usable_duration_s']))}"
        )
        print(f"  Saved to: {summary['output']}")

    if not summaries:
        print("No files processed", file=sys.stderr)
        sys.exit(1)

    total_windows = sum(int(s["num_windows"]) for s in summaries)
    total_usable_samples = sum(int(s["usable_unique_samples"]) for s in summaries)
    total_usable_duration_s = total_usable_samples / SAMPLE_RATE
    print(f"\n{'=' * 72}")
    print(f"Processed {len(summaries)} file(s), {total_windows} window(s) total")
    print(
        f"Total usable data (overlap removed): {total_usable_samples} samples"
        f" = {format_duration(total_usable_duration_s)}"
    )


if __name__ == "__main__":
    main()
