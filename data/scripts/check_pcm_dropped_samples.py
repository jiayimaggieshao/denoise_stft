#!/usr/bin/env python3
"""Check PCM txt files for dropped samples (idx gaps or timestamp anomalies)."""
# python data/scripts/check_pcm_dropped_samples.py
# ============ Run configuration (used when executing the script directly) ============
# Input txt: data/dataset/raw/ (DEFAULT_RAW_DIR in pcm_paths.py)

# File stem(s) under raw/, e.g. "heart_bw1" or ["heart_bw1", "noise2"]
SELECT_FILES = "heart_bw1"

# Use preprocessed txt from raw/{stem}_preprocessed.txt
USE_PREPROCESSED = False
# ====================================================================================

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pcm_paths import (
    DEFAULT_DATASET_DIR as DATASET_DIR,
    DEFAULT_RAW_DIR as RAW_DIR,
    collect_txt_files,
    resolve_input_paths,
)

SAMPLE_INTERVAL_US = 250  # 4 kHz
EXPECTED_HEADER = ("idx", "mono_us", "wall_epoch_us", "x")


@dataclass
class Gap:
    row_number: int
    prev_idx: int
    curr_idx: int
    missing_count: int


@dataclass
class TimestampMismatch:
    row_number: int
    column: str
    prev_value: int
    curr_value: int
    diff_us: int


@dataclass
class FileReport:
    path: Path
    total_rows: int = 0
    first_idx: int | None = None
    last_idx: int | None = None
    idx_gaps: list[Gap] = field(default_factory=list)
    mono_mismatches: list[TimestampMismatch] = field(default_factory=list)
    wall_mismatches: list[TimestampMismatch] = field(default_factory=list)
    error: str | None = None

    @property
    def missing_samples(self) -> int:
        return sum(gap.missing_count for gap in self.idx_gaps)

    @property
    def is_clean(self) -> bool:
        return (
            self.error is None
            and not self.idx_gaps
            and not self.mono_mismatches
            and not self.wall_mismatches
        )


def check_pcm_file(
    input_path: Path,
    *,
    check_timestamps: bool = True,
) -> FileReport:
    report = FileReport(path=input_path)

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            report.error = "File is empty"
            return report

        col_names = tuple(h.strip() for h in header)
        if col_names != EXPECTED_HEADER:
            report.error = f"Header mismatch: expected {EXPECTED_HEADER}, got {col_names}"
            return report

        prev_idx: int | None = None
        prev_mono: int | None = None
        prev_wall: int | None = None

        for row_number, row in enumerate(reader, start=2):
            if len(row) < 4:
                report.error = f"Row {row_number} has too few columns"
                return report

            try:
                idx = int(row[0].strip())
                mono_us = int(row[1].strip())
                wall_epoch_us = int(row[2].strip())
            except ValueError as exc:
                report.error = f"Failed to parse row {row_number}: {exc}"
                return report

            report.total_rows += 1
            if report.first_idx is None:
                report.first_idx = idx
            report.last_idx = idx

            if prev_idx is not None:
                diff = idx - prev_idx
                if diff != 1:
                    report.idx_gaps.append(
                        Gap(
                            row_number=row_number,
                            prev_idx=prev_idx,
                            curr_idx=idx,
                            missing_count=max(diff - 1, 0),
                        )
                    )

                if check_timestamps:
                    mono_diff = mono_us - prev_mono  # type: ignore[operator]
                    if mono_diff != SAMPLE_INTERVAL_US:
                        report.mono_mismatches.append(
                            TimestampMismatch(
                                row_number=row_number,
                                column="mono_us",
                                prev_value=prev_mono,  # type: ignore[arg-type]
                                curr_value=mono_us,
                                diff_us=mono_diff,
                            )
                        )

                    wall_diff = wall_epoch_us - prev_wall  # type: ignore[operator]
                    if wall_diff != SAMPLE_INTERVAL_US:
                        report.wall_mismatches.append(
                            TimestampMismatch(
                                row_number=row_number,
                                column="wall_epoch_us",
                                prev_value=prev_wall,  # type: ignore[arg-type]
                                curr_value=wall_epoch_us,
                                diff_us=wall_diff,
                            )
                        )

            prev_idx = idx
            prev_mono = mono_us
            prev_wall = wall_epoch_us

    if report.total_rows == 0:
        report.error = "No data rows found"

    return report


def print_report(report: FileReport, *, verbose: bool, max_issues: int) -> None:
    print(f"\n{'=' * 72}")
    print(f"File: {report.path}")

    if report.error:
        print(f"  [ERROR] {report.error}")
        return

    expected_span = (
        report.last_idx - report.first_idx + 1
        if report.first_idx is not None and report.last_idx is not None
        else 0
    )
    print(f"  Sample rows: {report.total_rows}")
    print(f"  idx range: {report.first_idx} .. {report.last_idx} (expected contiguous span: {expected_span})")

    if report.is_clean:
        print("  Result: no dropped samples, timestamp spacing OK")
        return

    if report.idx_gaps:
        print(
            f"  [idx gaps] {len(report.idx_gaps)} discontinuities, "
            f"{report.missing_samples} missing samples"
        )
        for gap in report.idx_gaps[:max_issues]:
            print(
                f"    row {gap.row_number}: idx {gap.prev_idx} -> {gap.curr_idx}"
                f" (missing {gap.missing_count})"
            )
        if len(report.idx_gaps) > max_issues:
            print(f"    ... {len(report.idx_gaps) - max_issues} more not shown")
    else:
        print("  [idx] contiguous, no dropped samples")

    if report.mono_mismatches:
        print(f"  [mono_us] {len(report.mono_mismatches)} intervals != {SAMPLE_INTERVAL_US} us")
        if verbose:
            for item in report.mono_mismatches[:max_issues]:
                print(
                    f"    row {item.row_number}: {item.prev_value} -> {item.curr_value}"
                    f" (delta={item.diff_us} us)"
                )

    if report.wall_mismatches:
        print(f"  [wall_epoch_us] {len(report.wall_mismatches)} intervals != {SAMPLE_INTERVAL_US} us")
        if verbose:
            for item in report.wall_mismatches[:max_issues]:
                print(
                    f"    row {item.row_number}: {item.prev_value} -> {item.curr_value}"
                    f" (delta={item.diff_us} us)"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check PCM txt for dropped samples (idx gaps or 4 kHz timestamp anomalies)"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input file(s) or directory (default: SELECT_FILES at top of script)",
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help=f"Use preprocessed txt (default: USE_PREPROCESSED = {USE_PREPROCESSED})",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search for raw txt files in directory mode",
    )
    parser.add_argument(
        "--no-timestamp-check",
        action="store_true",
        help="Check idx continuity only, skip mono_us / wall_epoch_us spacing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print timestamp mismatch details",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=20,
        help="Max issues to display per file (default: 20)",
    )
    args = parser.parse_args()
    use_preprocessed = args.preprocessed or USE_PREPROCESSED

    if args.inputs:
        files = collect_txt_files(
            args.inputs, recursive=args.recursive, use_preprocessed=use_preprocessed
        )
    else:
        files = resolve_input_paths(SELECT_FILES, DATASET_DIR, use_preprocessed=use_preprocessed)
        print("Using script configuration:")
        print(f"  RAW_DIR          = {RAW_DIR}")
        print(f"  SELECT_FILES     = {SELECT_FILES}")
        print(f"  USE_PREPROCESSED = {use_preprocessed}")
        for file_path in files:
            print(f"  -> {file_path}")

    if not files:
        print("No input files found", file=sys.stderr)
        sys.exit(1)

    reports = [
        check_pcm_file(
            file_path,
            check_timestamps=not args.no_timestamp_check,
        )
        for file_path in files
    ]

    clean_count = 0
    for report in reports:
        print_report(report, verbose=args.verbose, max_issues=args.max_issues)
        if report.is_clean:
            clean_count += 1

    print(f"\n{'=' * 72}")
    print(
        f"Checked {len(reports)} file(s): {clean_count} passed, "
        f"{len(reports) - clean_count} with issues"
    )

    has_failure = any(not report.is_clean for report in reports)
    sys.exit(1 if has_failure else 0)


if __name__ == "__main__":
    main()
