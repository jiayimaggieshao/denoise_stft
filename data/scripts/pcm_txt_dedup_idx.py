#!/usr/bin/env python3
"""Remove duplicate idx rows from PCM txt (keep first occurrence)."""

# python data/scripts/pcm_txt_dedup_idx.py
# ============ Run configuration (used when executing the script directly) ============
# Input txt: data/dataset/raw/ (DEFAULT_RAW_DIR in pcm_paths.py)

# File stem(s) under raw/, e.g. "heart_aw2" or ["heart_aw2", "heart_aw3"]
SELECT_FILES = "heart_aw2"

# When True, overwrite the input txt in place (via a temp file).
OVERWRITE_INPUT = True
# ====================================================================================

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pcm_paths import (
    DEFAULT_DATASET_DIR as DATASET_DIR,
    DEFAULT_RAW_DIR as RAW_DIR,
    resolve_preprocess_jobs,
)

EXPECTED_HEADER = ("idx", "mono_us", "wall_epoch_us", "x")


def dedup_pcm_txt(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
) -> dict[str, int | str]:
    seen_idx: set[int] = set()
    rows_in = 0
    rows_out = 0
    duplicates = 0
    x_mismatches = 0
    first_mismatch: tuple[int, int, int] | None = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with input_path.open(newline="", encoding="utf-8") as infile, temp_path.open(
        "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        col_names = tuple(h.strip() for h in header)
        if col_names != EXPECTED_HEADER:
            raise ValueError(f"Header mismatch: expected {EXPECTED_HEADER}, got {col_names}")
        writer.writerow(EXPECTED_HEADER)

        idx_to_x: dict[int, int] = {}

        for row_number, row in enumerate(reader, start=2):
            if len(row) < 4:
                raise ValueError(f"Row {row_number} has too few columns")

            rows_in += 1
            idx = int(row[0].strip())
            x = int(row[3].strip())

            if idx in seen_idx:
                duplicates += 1
                if idx in idx_to_x and idx_to_x[idx] != x:
                    x_mismatches += 1
                    if first_mismatch is None:
                        first_mismatch = (idx, idx_to_x[idx], x)
                continue

            seen_idx.add(idx)
            idx_to_x[idx] = x
            writer.writerow(row)
            rows_out += 1

    if x_mismatches:
        temp_path.unlink(missing_ok=True)
        detail = ""
        if first_mismatch is not None:
            idx, kept_x, dup_x = first_mismatch
            detail = f" (example idx {idx}: kept x={kept_x}, duplicate x={dup_x})"
        raise ValueError(
            f"Found {x_mismatches} duplicate idx with differing x values{detail}; "
            "aborting without writing output"
        )

    if overwrite and output_path.resolve() == input_path.resolve():
        temp_path.replace(input_path)
        final_path = input_path
    else:
        temp_path.replace(output_path)
        final_path = output_path

    return {
        "input": str(input_path),
        "output": str(final_path),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "duplicates_removed": duplicates,
        "unique_idx": rows_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove duplicate idx rows from PCM txt (keep first occurrence)."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input txt file(s). Default: SELECT_FILES under RAW_DIR",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Dataset root (default: {DATASET_DIR}; input txt under {{dataset-dir}}/raw/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output txt path (single input only). Default: overwrite input.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite input; write sibling *_deduped.txt instead.",
    )
    args = parser.parse_args()

    overwrite = OVERWRITE_INPUT and not args.no_overwrite

    if args.inputs:
        jobs: list[tuple[str, Path, Path]] = []
        for input_path in args.inputs:
            resolved = input_path.resolve()
            if args.output is not None and len(args.inputs) != 1:
                raise SystemExit("--output requires exactly one input file")
            if args.output is not None:
                out_path = args.output.resolve()
            elif overwrite:
                out_path = resolved
            else:
                out_path = resolved.with_name(f"{resolved.stem}_deduped.txt")
            jobs.append((resolved.stem, resolved, out_path))
    else:
        jobs = []
        for stem, input_path, _output_path in resolve_preprocess_jobs(
            SELECT_FILES, args.dataset_dir
        ):
            if args.output is not None:
                raise SystemExit("--output requires explicit input paths")
            if overwrite:
                out_path = input_path
            else:
                out_path = input_path.with_name(f"{input_path.stem}_deduped.txt")
            jobs.append((stem, input_path, out_path))

        print("Using script configuration:")
        print(f"  RAW_DIR        = {RAW_DIR}")
        print(f"  SELECT_FILES   = {SELECT_FILES}")
        print(f"  OVERWRITE_INPUT = {overwrite}")

    summaries: list[dict[str, int | str]] = []
    for _stem, input_path, output_path in jobs:
        if not input_path.is_file():
            print(f"[SKIP] File not found: {input_path}", file=sys.stderr)
            continue

        print(f"\nProcessing: {input_path}")
        summary = dedup_pcm_txt(input_path, output_path, overwrite=overwrite)
        summaries.append(summary)
        print(f"  Rows in:  {summary['rows_in']}")
        print(f"  Rows out: {summary['rows_out']}")
        print(f"  Duplicates removed: {summary['duplicates_removed']}")
        print(f"  Saved to: {summary['output']}")

    if not summaries:
        print("No files processed", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 72}")
    print(f"Processed {len(summaries)} file(s)")


if __name__ == "__main__":
    main()
