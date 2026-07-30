# Repository and Data Audit

Audit date: 2026-07-23

## Executive finding

The existing waveform CleanUNet experiment is not a reliable reference implementation for the wearable two-microphone setting. It discards the exterior microphone, trains on an easier synthetic distribution than the real walking recordings, and reconstructs `step_0.1s` inference windows with a default one-second hop. The archive also contains no trained checkpoint, so the earlier waveform result cannot be reproduced exactly from the supplied files.

The new implementation keeps the old path intact for comparison and adds a separate frequency-domain path (`CardioSpecNet`) that uses a complex STFT mask conditioned on exterior-microphone magnitude, plus a complex transfer canceller and calibrated neural–adaptive fusion. It includes on-the-fly transfer-path augmentation, subject/recording-level held-out evaluation, exact timestamp alignment, single-channel fallback, raw-versus-band-limited target sensitivity, and real-data proxy checks.

## Dataset inventory

All supplied windows contain 8,000 samples at 4,000 Hz (2.0 seconds).

| Split | Recordings observed |
|---|---|
| Clean train | `heart_aw1`, `heart_aw2`, `heart_aw6`, `heart_bw1`, `heart_bw2`, `heart_bw5`, `heart_bw6` |
| Clean validation | `heart_aw4`, `heart_bw4` |
| Exterior-noise train | `noise1`, `noise2`, `noise3`, `noise5` |
| Exterior-noise validation | `noise6` |
| Real walking chest | `heart_w1` through `heart_w6` |

The 0.1-second source step creates 95% overlap between adjacent 2-second windows. Treating all stored windows as independent inflates sample count and encourages memorization. The new dataset loader takes every tenth source window by default, preserving the requested 0.1-second archive while using an effective one-second source stride for training.

### Real chest/reference pairing

The wall-clock timestamps show that `heart_w2`, `heart_w3`, `heart_w5`, and `heart_w6` overlap their corresponding `noise2`, `noise3`, `noise5`, and `noise6` recordings. Subjects 1 and 4 do not have a usable synchronized exterior reference in the archive.

Raw waveform correlation is weak (median absolute correlation approximately 0.056; median magnitude-squared coherence near 0.01 in the heart bands), which rules out assuming phase coherence. Log-band energy correlation is nevertheless useful and subject-dependent. Examples include 0.75 in 15–40 Hz for subject 5 and 0.77 in 400–800 Hz for subject 6. This supports magnitude conditioning, but not direct sample-by-sample subtraction.

### Spectral observations

Median aggregate spectral energy fractions from the supplied recordings:

| Band (Hz) | Clean sitting | Exterior noise | Real walking chest |
|---:|---:|---:|---:|
| 0–15 | 21.8% | 57.2% | 86.1% |
| 15–40 | 55.2% | 8.3% | 7.0% |
| 40–100 | 8.5% | 1.4% | 0.4% |
| 200–400 | 0.18% | 3.6% | 0.09% |
| 400–800 | 0.18% | 13.0% | 0.13% |

The strong walking energy below 15 Hz makes a high-pass transition indispensable. A hard low-pass near 200 Hz would be unsafe, however, because pathological murmurs can extend to roughly 600–800 Hz. The implemented passband is therefore 15–800 Hz, with smooth transitions to avoid ringing.

The machine-readable analysis files are under `analysis_outputs/`.

## Problems found in the original pipeline

### 1. The exterior microphone is unused

The deployed hardware is explicitly two-channel, yet `src/model.py`, `src/dataset.py`, and `scripts/run_inference.py` accept only one waveform. This loses the strongest available indicator of environmental/motion contamination.

### 2. Synthetic training is too easy relative to deployment

`augment_data.py` forms `clean + scaled exterior_noise` with the same noise waveform that conceptually serves as the exterior reference. In the real device, clothing, body transmission, microphone frequency response, delay, organ sounds, and contact artifacts make the chest-noise path different from the exterior channel. The new mixer randomizes delay, impulse response, equalization, polarity, reference gain, heart leakage, transient rubbing, and an independent-noise component.

### 3. Augmentation can break pair consistency

`src/dataset.py` applies an outer gain/polarity to both signals, then calls the configurable augmenter only on the noisy signal. With the current Gaussian-noise-only setup that is defensible, but enabling time shift, gain, or polarity inside the augmenter would transform only the input or apply a transform twice. The new mixer constructs the paired input/target together in one deterministic operation.

### 4. Pre-generating one file per pair wastes storage and I/O

The old pipeline materializes thousands of highly overlapping `.npy` files. The new dataset samples from overlap-thinned NPZ pools and mixes in memory, so each epoch sees a new contamination path without duplicating the source data.

### 5. README defaults do not match executable defaults

The README describes an approximately 40-million-parameter CleanUNet (`H=64`, depth 8, three transformer layers, model width 512) and SI-SNR as the primary loss. The actual dataclass defaults are `H=32`, depth 6, one transformer layer, width 256. The actual loss defaults disable SI-SNR (`0.0`) and use multi-resolution STFT weight `2.0` plus waveform L1 weight `1.0`. This configuration drift makes reported experiments ambiguous.

### 6. Existing STFT loss resolutions are too coarse for this sample rate

The old loss uses windows of 1,024, 2,048, and 4,096 samples—256 ms, 512 ms, and 1.024 s at 4 kHz. Those scales are useful for slow spectral structure but can blur S1/S2 transients. The new primary STFT uses a 256-sample (64 ms) Hann window, 64-sample (16 ms) hop, and 512-point FFT.

### 7. Inference reconstructs 0.1-second-hop data with a one-second default

`scripts/run_inference.py` defaults `--step` to one second, while the documented command points to `step_0.1s` data and does not override the flag. This stretches the output timeline by approximately 10× and changes overlap averaging. It also ignores stored `start_idx`, `segment_id`, and wall-clock timestamps. The replacement script uses exact sample indices and timestamp-aligns the exterior reference.

### 8. Inference normalizes windows redundantly

The old script RMS-normalizes each window before a model that already performs internal per-utterance normalization. This alters behavior relative to training and can make adjacent overlapping windows inconsistent. CardioSpecNet performs one internal chest-derived scaling and restores the physical scale at its output.

### 9. Evaluation is insufficient for a diagnostic signal

A single aggregate waveform loss or SI-SDR can improve while murmur bands are attenuated. The new benchmark reports scale-dependent SNR, SI-SDR, correlation, log-spectral distance, and relative error in 15–40, 40–100, 100–200, 200–400, and 400–800 Hz. The loss also includes an asymmetric over-attenuation term and per-band log-energy preservation.

### 10. Real walking data have no clean ground truth

No objective score on `heart_w*` can prove diagnostic preservation. The real-data script reports only explicitly labeled proxy metrics: sub-15-Hz suppression, output/input band ratios, and change in correlation with the exterior reference. Clinical listening, pathology labels, and downstream detection performance remain necessary before deployment.

### 11. The target definition was previously implicit

A model trained to output a 15–800 Hz monitoring signal is not solving exactly the same objective as reconstructing the unfiltered quiet waveform `X_t`. The new evaluator exposes `--target_mode bandlimited` and `--target_mode raw`, and the benchmark reports both. The target sensitivity materially changes SI-SDR and the preferred fusion weight, so any future experiment must name the target construction explicitly.

## Positive aspects retained

The repository already uses participant-separated clean validation (subject 4) and a held-out noise recording (subject 6), provides wall-clock and sample-index metadata, and stores several source steps. Those choices enable a substantially stronger evaluation without redesigning the data format.
