"""Path and file-selection helpers for the PCM dataset layout."""

from pathlib import Path

DATASET_DIR_NAME = "dataset"
RAW_DIR_NAME = "raw"
AUDIO_DIR_NAME = "audio"
WINDOWS_DIR_NAME = "windows"

_DATA_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = _DATA_ROOT / DATASET_DIR_NAME
DEFAULT_RAW_DIR = DEFAULT_DATASET_DIR / RAW_DIR_NAME
DEFAULT_AUDIO_DIR = DEFAULT_DATASET_DIR / AUDIO_DIR_NAME
DEFAULT_WINDOWS_DIR = DEFAULT_DATASET_DIR / WINDOWS_DIR_NAME


def data_root(script_file: Path | str | None = None) -> Path:
    """Return the `data/` directory (parent of `scripts/`)."""
    if script_file is None:
        return _DATA_ROOT
    return Path(script_file).resolve().parent.parent


def default_dataset_dir(script_file: Path | str | None = None) -> Path:
    """Return `data/dataset/` — local working dir for raw/audio/windows."""
    if script_file is None:
        return DEFAULT_DATASET_DIR
    return data_root(script_file) / DATASET_DIR_NAME


def default_raw_dir(script_file: Path | str | None = None) -> Path:
    """Return `data/dataset/raw/` — input PCM txt directory."""
    return raw_dir(default_dataset_dir(script_file))


def normalize_selection(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def raw_dir(dataset_dir: Path | str) -> Path:
    return Path(dataset_dir) / RAW_DIR_NAME


def audio_dir(dataset_dir: Path | str) -> Path:
    return Path(dataset_dir) / AUDIO_DIR_NAME


def windows_dir(dataset_dir: Path | str) -> Path:
    return Path(dataset_dir) / WINDOWS_DIR_NAME


def step_windows_subdir(step_seconds: float) -> str:
    """Subfolder under windows/, e.g. step_0.01s, step_0.1s, step_1s."""
    if step_seconds == int(step_seconds):
        return f"step_{int(step_seconds)}s"
    return f"step_{step_seconds}s"


def windows_dir_for_step(dataset_dir: Path | str, step_seconds: float) -> Path:
    return windows_dir(dataset_dir) / step_windows_subdir(step_seconds)


def file_stem_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_preprocessed"):
        return stem.removesuffix("_preprocessed")
    return stem


def input_txt_name(stem: str, *, use_preprocessed: bool) -> str:
    if use_preprocessed:
        return f"{stem}_preprocessed.txt"
    return f"{stem}.txt"


def output_stem(stem: str, *, use_preprocessed: bool) -> str:
    if use_preprocessed:
        return f"{stem}_preprocessed"
    return stem


def resolve_input_txt(
    stem: str,
    dataset_dir: Path | str,
    *,
    use_preprocessed: bool,
) -> Path:
    return raw_dir(dataset_dir) / input_txt_name(stem, use_preprocessed=use_preprocessed)


def resolve_preprocessed_txt(stem: str, dataset_dir: Path | str) -> Path:
    return raw_dir(dataset_dir) / f"{stem}_preprocessed.txt"


def resolve_wav_output(
    stem: str,
    dataset_dir: Path | str,
    *,
    use_preprocessed: bool,
) -> Path:
    name = f"{output_stem(stem, use_preprocessed=use_preprocessed)}.wav"
    return audio_dir(dataset_dir) / name


def resolve_windows_output(
    stem: str,
    dataset_dir: Path | str,
    *,
    use_preprocessed: bool,
    step_seconds: float,
) -> Path:
    name = f"{output_stem(stem, use_preprocessed=use_preprocessed)}_windows.npz"
    return windows_dir_for_step(dataset_dir, step_seconds) / name


def resolve_input_paths(
    stems: str | list[str],
    dataset_dir: Path | str,
    *,
    use_preprocessed: bool,
) -> list[Path]:
    return [
        resolve_input_txt(stem, dataset_dir, use_preprocessed=use_preprocessed)
        for stem in normalize_selection(stems)
    ]


def resolve_preprocess_jobs(
    stems: str | list[str],
    dataset_dir: Path | str,
) -> list[tuple[str, Path, Path]]:
    jobs: list[tuple[str, Path, Path]] = []
    for stem in normalize_selection(stems):
        input_path = resolve_input_txt(stem, dataset_dir, use_preprocessed=False)
        output_path = resolve_preprocessed_txt(stem, dataset_dir)
        jobs.append((stem, input_path, output_path))
    return jobs


def resolve_wav_jobs(
    stems: str | list[str],
    dataset_dir: Path | str,
    *,
    use_preprocessed: bool,
) -> list[tuple[str, Path, Path]]:
    jobs: list[tuple[str, Path, Path]] = []
    for stem in normalize_selection(stems):
        input_path = resolve_input_txt(stem, dataset_dir, use_preprocessed=use_preprocessed)
        output_path = resolve_wav_output(stem, dataset_dir, use_preprocessed=use_preprocessed)
        jobs.append((stem, input_path, output_path))
    return jobs


def resolve_windows_jobs(
    stems: str | list[str],
    dataset_dir: Path | str,
    *,
    use_preprocessed: bool,
    step_seconds: float,
) -> list[tuple[str, Path, Path]]:
    jobs: list[tuple[str, Path, Path]] = []
    for stem in normalize_selection(stems):
        input_path = resolve_input_txt(stem, dataset_dir, use_preprocessed=use_preprocessed)
        output_path = resolve_windows_output(
            stem, dataset_dir, use_preprocessed=use_preprocessed, step_seconds=step_seconds
        )
        jobs.append((stem, input_path, output_path))
    return jobs


def collect_txt_files(
    paths: list[Path],
    recursive: bool,
    *,
    use_preprocessed: bool,
) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved.is_file():
            if resolved.suffix.lower() == ".txt":
                files.append(resolved)
            continue
        if not resolved.is_dir():
            raise FileNotFoundError(f"Path does not exist: {resolved}")

        pattern = "**/*" if recursive else "*"
        for candidate in sorted(resolved.glob(pattern)):
            if not candidate.is_file() or candidate.suffix.lower() != ".txt":
                continue
            if use_preprocessed:
                if candidate.stem.endswith("_preprocessed"):
                    files.append(candidate)
            elif not candidate.stem.endswith("_preprocessed"):
                files.append(candidate)

    seen: set[Path] = set()
    unique_files: list[Path] = []
    for file_path in files:
        if file_path not in seen:
            seen.add(file_path)
            unique_files.append(file_path)
    return unique_files


def wav_output_for_input(input_path: Path, dataset_dir: Path | str) -> Path:
    stem = file_stem_from_path(input_path)
    use_preprocessed = input_path.stem.endswith("_preprocessed")
    return resolve_wav_output(stem, dataset_dir, use_preprocessed=use_preprocessed)


def windows_output_for_input(
    input_path: Path,
    dataset_dir: Path | str,
    *,
    step_seconds: float,
) -> Path:
    stem = file_stem_from_path(input_path)
    use_preprocessed = input_path.stem.endswith("_preprocessed")
    return resolve_windows_output(
        stem, dataset_dir, use_preprocessed=use_preprocessed, step_seconds=step_seconds
    )


def preprocess_output_for_input(input_path: Path, dataset_dir: Path | str) -> Path:
    stem = file_stem_from_path(input_path)
    return resolve_preprocessed_txt(stem, dataset_dir)
