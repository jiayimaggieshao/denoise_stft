# Data

Windowed heart-sound npz (4 kHz, **2 s window** / **8000 samples**, step chosen at cut time).

Step subfolders (`step_0.01s`, `step_0.1s`, `step_1s`) hold the same participants; only sliding-window density differs.

## Layout

```
data/
  clean/
    step_0.01s/
      train/          # quiet heart sounds — train split
      val/            # quiet heart sounds — val split
    step_0.1s/
      train/
      val/
    step_1s/
      train/
      val/
  noise/
    step_0.01s/
      train/          # motion noise — train split
      val/
    step_0.1s/
      train/
      val/
    step_1s/
      train/
      val/
  test_real/walking/
    step_0.01s/       # walking heart sounds — held-out eval
    step_0.1s/
    step_1s/
  dataset/            # local working dir (not in git)
  scripts/            # PCM txt → wav / npz tools
```

## clean/

Quiet recordings (`heart_bw*`, `heart_aw*`). Each participant has raw (`*_windows.npz`) and preprocessed (`*_preprocessed_windows.npz`) variants. Preprocessed = DC removal + 20–500 Hz bandpass before windowing.

| Split | Files |
|-------|-------|
| `train/` | `heart_bw1`, `heart_bw2`, `heart_bw5`, `heart_bw6`, `heart_aw1`, `heart_aw2`, `heart_aw6` |
| `val/` | `heart_bw4`, `heart_aw4` |

Same files under every `step_*` folder.

| Prefix | Description |
|--------|-------------|
| `heart_bw*` | Before walking |
| `heart_aw*` | After walking |

## noise/

Pure ambient noise during walking.

| Split | Files |
|-------|-------|
| `train/` | `noise1`, `noise2`, `noise3`, `noise5` |
| `val/` | `noise6` |

Same files under every `step_*` folder.

## test_real/walking/

Real walking heart sounds (not used by `augment_data.py`): `heart_w1`–`heart_w6`, under each `step_*` subfolder.

## NPZ format

| Key | Shape | Description |
|-----|-------|-------------|
| `x` | `(N, 8000)` int16 | Window samples |
| `start_idx` | `(N,)` int64 | idx of first sample |
| `segment_id` | `(N,)` int32 | Continuous idx segment |
| `start_wall_epoch_us` | `(N,)` int64 | Wall-clock timestamp |

## augment_data.py

Edit the config block at the top of `augment_data.py`, then run:

```bash
python augment_data.py
```

| Setting | Options | Meaning |
|---------|---------|---------|
| `WINDOW_STEP_SECONDS` | `0.01`, `0.1`, `1.0` | Read from `clean/step_*` and `noise/step_*` (e.g. `0.1` → `step_0.1s`) |
| `SPLIT` | `"train"`, `"val"`, `"both"` | Which splits to synthesise (default: both) |
| `N_TRAIN` / `N_VAL` | e.g. `10000` / `1000` | Synthetic pairs per split (random sample from pool) |
| `USE_PREPROCESSED` | `True` / `False` | Target clean type |

CLI override: `python augment_data.py --step-seconds 0.01`

**Mixing modes** (SNR from `X_raw` vs `N_raw`):

- `USE_PREPROCESSED = False` → **X_raw + α·N_raw → X_raw**
- `USE_PREPROCESSED = True` → **X_raw + α·N_raw → X_pre** (same window index)

Outputs are RMS-normalized (`Y/g`, `target/g`) before saving to `synthetic_data/{train,val}/`.

## Scripts (local preprocessing)

Scripts read/write `data/dataset/` by default. Edit `SELECT_FILES` at the top of each script:

```bash
python data/scripts/pcm_txt_to_windows.py
```

| Script | Purpose |
|--------|---------|
| `check_pcm_dropped_samples.py` | Report idx gaps in raw/preprocessed txt |
| `pcm_txt_preprocess.py` | DC removal + bandpass → `{stem}_preprocessed.txt` |
| `pcm_txt_to_wav.py` | txt → WAV in `dataset/audio/` |
| `pcm_txt_to_windows.py` | txt → windowed npz in `dataset/windows/step_*/` (set `WINDOW_STEP_SECONDS` at top) |
| `pcm_txt_dedup_idx.py` | Remove duplicate idx rows |
| `pcm_paths.py` | Shared path helpers |

To regenerate npz locally, create `data/dataset/`:

```
dataset/
  raw/              # PCM txt (header: idx, mono_us, wall_epoch_us, x)
  audio/            # optional WAV exports
  windows/
    step_0.01s/     # 2 s window, 0.01 s step
    step_0.1s/      # 2 s window, 0.1 s step
    step_1s/        # 2 s window, 1 s step
```

Copy finished npz into `clean/step_*/{train,val}/`, `noise/step_*/{train,val}/`, or `test_real/walking/step_*/` as appropriate.

Some source files (raw txt, wav) are on [Google Drive](https://drive.google.com/drive/folders/1VoEV9ZqUG_l850LAx18aW93ZHuZbxY4B).
