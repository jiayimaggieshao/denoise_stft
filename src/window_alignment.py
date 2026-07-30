"""Timestamp alignment and overlap-add reconstruction for windowed PCG archives."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WindowArchive:
    x: np.ndarray
    start_idx: np.ndarray
    segment_id: np.ndarray
    wall_time_us: np.ndarray


def load_window_archive(path: Path | str | None) -> WindowArchive | None:
    if path is None:
        return None
    archive_path = Path(path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Window archive not found: {archive_path}")
    payload = np.load(archive_path)
    if "x" not in payload:
        raise KeyError(f"{archive_path} must contain key 'x'")
    wall_time = payload["start_wall_epoch_us"] if "start_wall_epoch_us" in payload else payload.get(
        "wall_time_us"
    )
    if wall_time is None:
        raise KeyError(f"{archive_path} must contain start_wall_epoch_us or wall_time_us")
    start_idx = payload["start_idx"] if "start_idx" in payload else np.arange(len(payload["x"]), dtype=np.int64)
    segment_id = payload["segment_id"] if "segment_id" in payload else np.zeros(len(payload["x"]), dtype=np.int32)
    return WindowArchive(
        x=payload["x"],
        start_idx=start_idx.astype(np.int64),
        segment_id=segment_id.astype(np.int32),
        wall_time_us=wall_time.astype(np.int64),
    )


def nearest_timestamp_alignment(
    target_times_us: np.ndarray,
    reference_times_us: np.ndarray,
    max_delta_us: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest reference index and availability mask for each target timestamp."""
    target_times_us = np.asarray(target_times_us, dtype=np.int64)
    reference_times_us = np.asarray(reference_times_us, dtype=np.int64)
    if reference_times_us.size == 0:
        return np.zeros(len(target_times_us), dtype=np.int64), np.zeros(len(target_times_us), dtype=bool)

    insertion = np.searchsorted(reference_times_us, target_times_us)
    candidates = np.stack(
        [
            np.clip(insertion - 1, 0, len(reference_times_us) - 1),
            np.clip(insertion, 0, len(reference_times_us) - 1),
        ],
        axis=1,
    )
    left = reference_times_us[candidates[:, 0]]
    right = reference_times_us[candidates[:, 1]]
    choose_right = np.abs(right - target_times_us) <= np.abs(left - target_times_us)
    indices = np.where(choose_right, candidates[:, 1], candidates[:, 0])
    deltas = np.abs(reference_times_us[indices] - target_times_us)
    available = deltas <= max_delta_us
    return indices.astype(np.int64), available.astype(bool)


def overlap_add_windows(
    windows: np.ndarray,
    starts: np.ndarray,
    synthesis_window: str = "hann",
) -> np.ndarray:
    """Reconstruct a continuous waveform from possibly overlapping windows."""
    windows = np.asarray(windows, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.int64)
    if windows.ndim != 2:
        raise ValueError(f"windows must be [N, T], got {windows.shape}")
    if len(starts) != len(windows):
        raise ValueError("starts length must match number of windows")

    window_length = windows.shape[1]
    end = int(starts.max()) + window_length
    output = np.zeros(end, dtype=np.float32)
    weights = np.zeros(end, dtype=np.float32)

    if synthesis_window == "ones":
        weight = np.ones(window_length, dtype=np.float32)
    elif synthesis_window == "hann":
        weight = np.hanning(window_length).astype(np.float32)
    else:
        raise ValueError(f"Unsupported synthesis_window: {synthesis_window}")

    for window, start in zip(windows, starts, strict=True):
        stop = int(start) + window_length
        output[int(start):stop] += window * weight
        weights[int(start):stop] += weight

    nonzero = weights > 0
    output[nonzero] /= weights[nonzero]
    return output
