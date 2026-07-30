# Benchmark Results

Benchmark date: 2026-07-23

## What is being measured

The synthetic benchmark draws quiet chest windows from `data/clean/step_0.1s/val` (subject 4) and exterior-noise windows from `data/noise/step_0.1s/val` (subject 6). Source windows are thinned by ten before sampling so the original 95%-overlapped archive is not treated as independent data. The main evaluation contains 512 deterministic two-second mixtures, seed 42424, with input SNR sampled uniformly from -10 to +10 dB.

There are two target definitions, and the distinction is important:

- **Band-limited target**: the quiet recording after the same smooth 5→15 Hz and 800→1,000 Hz transitions used for model training. This tests a denoising-plus-band-limiting monitoring output.
- **Raw target**: the unfiltered quiet recording supplied as `X_t`. This tests fidelity to every component of the original quiet waveform.

The monitoring hybrid uses 55% CardioSpecNet and 45% complex transfer cancellation. The raw-fidelity hybrid uses 20% CardioSpecNet and 80% complex transfer cancellation. Those weights were selected on a separate 256-mixture calibration run (seed 5151); the tables below use seed 42424. Calibration and evaluation still use the same held-out source recordings, so this is not a participant-independent external test.

## Band-limited monitoring target: exterior reference available

| Method | SNR improvement (dB) | SI-SDR improvement (dB) | Correlation | Log-spectral distance, 15–800 Hz (dB; lower is better) |
|---|---:|---:|---:|---:|
| Raw mixture | 0.000 | 0.000 | 0.470 | 11.088 |
| Soft bandpass only | 6.233 | 4.029 | 0.621 | 11.020 |
| Reference spectral subtraction | 8.599 | 4.694 | 0.647 | 10.447 |
| Complex transfer cancellation | 8.093 | 5.761 | 0.682 | 9.646 |
| CardioSpecNet | **9.374** | 5.249 | 0.667 | 10.435 |
| **Hybrid fusion (55% neural)** | 9.287 | **6.026** | **0.692** | **8.909** |
| Oracle Wiener (non-deployable) | 13.409 | 10.972 | 0.862 | 9.439 |

The hybrid is the recommended **reference-available monitoring profile**: it gives up only 0.087 dB of SNR improvement relative to CardioSpecNet while leading every deployable method on SI-SDR, correlation, and 15–800 Hz spectral distance. CardioSpecNet remains the strongest standalone method on scale-dependent SNR. Relative to bandpass alone, CardioSpecNet adds 3.141 dB SNR improvement; the rest of its headline 9.374 dB comes from the explicitly defined band-limiting operation.

The oracle is not guaranteed to lead log-spectral distance because it optimizes an idealized Wiener estimate, not that metric directly. It uses the clean target and cannot be deployed.

### Median relative band-energy error

Lower is better. A value of zero is exact band energy; one is an absolute error equal to the target energy.

| Method | 15–40 Hz | 40–100 Hz | 100–200 Hz | 200–400 Hz | 400–800 Hz |
|---|---:|---:|---:|---:|---:|
| Reference spectral subtraction | 0.388 | 0.182 | 0.633 | 0.954 | 1.914 |
| Complex transfer cancellation | 0.225 | 0.224 | 1.559 | 2.493 | 4.392 |
| CardioSpecNet | 0.357 | 0.175 | **0.505** | **0.666** | **0.847** |
| Hybrid fusion | **0.299** | **0.145** | 0.542 | 0.987 | 2.125 |
| Oracle Wiener (non-deployable) | 0.473 | 0.131 | 0.366 | 0.521 | 0.638 |

The neural model is substantially more conservative than complex cancellation in the synthetic 100–800 Hz target bands. Fusion improves global waveform metrics but partially inherits the adaptive method's high-band overestimation. This is why the repo exports the neural, adaptive, and fused waveforms separately.

## Exact raw quiet-waveform target

This sensitivity test evaluates the same mixtures against the original unfiltered quiet recording. The hybrid automatically uses the separately calibrated 20% neural weight.

| Method | SNR improvement (dB) | SI-SDR improvement (dB) | Correlation | Log-spectral distance, 15–800 Hz (dB; lower is better) |
|---|---:|---:|---:|---:|
| Raw mixture | 0.000 | 0.000 | 0.628 | 11.026 |
| Soft bandpass only | 2.451 | -0.728 | 0.607 | 11.015 |
| Reference spectral subtraction | 3.223 | -0.668 | 0.612 | 10.486 |
| Complex transfer cancellation | 3.511 | 0.828 | 0.664 | 9.650 |
| CardioSpecNet | 3.496 | -0.554 | 0.617 | 10.481 |
| **Hybrid fusion (20% neural)** | **3.658** | **0.852** | **0.666** | **9.272** |
| Oracle Wiener (non-deployable) | 8.429 | 7.251 | 0.866 | 9.364 |

The raw-target result is the most faithful interpretation of the user's equation `Y_t = X_t + ε_t`. It also exposes the cost of a fixed PCG bandpass: CardioSpecNet intentionally removes quiet out-of-band energy and therefore has negative raw-target SI-SDR improvement despite positive SNR improvement. For applications that require the original quiet waveform rather than a monitoring-band signal, use `--fusion_neural_weight 0.20` and retain the complex-transfer output for comparison.

## Missing-reference ablation

The exterior channel is zeroed for every example. Classical reference methods reduce to bandpass filtering.

| Method | SNR improvement (dB) | SI-SDR improvement (dB) | Correlation | LSD (dB) |
|---|---:|---:|---:|---:|
| Soft bandpass only | 6.233 | 4.029 | 0.621 | 11.020 |
| Reference spectral subtraction | 6.233 | 4.029 | 0.621 | 11.028 |
| Complex transfer cancellation | 6.233 | 4.029 | 0.621 | 11.020 |
| CardioSpecNet | **7.254** | **4.369** | **0.633** | **10.367** |
| Reference-aware hybrid fusion | **7.254** | **4.369** | **0.633** | **10.367** |

When the exterior channel is unavailable, the deployment fusion now falls back to CardioSpecNet window-by-window, so it matches the standalone neural result. Reference-dropout training gives that fallback 1.021 dB additional SNR improvement over filtering alone.

## Performance by input-SNR range

Each row contains 256 fresh deterministic mixtures. Cells are `SNR improvement / SI-SDR improvement` in dB. The hybrid uses the 55% monitoring weight.

| Input SNR | Bandpass | Ref. subtraction | Complex transfer | CardioSpecNet | Hybrid fusion | Oracle Wiener |
|---|---:|---:|---:|---:|---:|---:|
| -10 to -5 dB | 5.345 / 3.134 | 9.074 / 3.856 | 8.041 / 5.749 | **10.399** / 4.334 | 9.616 / 5.469 | 17.086 / 13.814 |
| -5 to 0 dB | 5.642 / 3.409 | 8.713 / 4.447 | 7.968 / 5.639 | **9.660** / 5.074 | 9.345 / **5.792** | 14.124 / 11.547 |
| 0 to +5 dB | 6.310 / 4.083 | 8.322 / 4.952 | 7.987 / 5.641 | 8.887 / 5.566 | **9.043 / 6.126** | 12.020 / 9.880 |
| +5 to +10 dB | 7.354 / 5.173 | 8.072 / 5.373 | 8.098 / 5.737 | 8.428 / 5.849 | **8.858 / 6.412** | 10.890 / 8.965 |

At severe contamination, CardioSpecNet maximizes scale-dependent SNR while the adaptive method can retain better SI-SDR. Fusion becomes the strongest deployable method on both metrics from 0 dB upward and provides the best overall compromise across the full range.

## Real walking recordings: proxy metrics only

Subjects 2, 3, 5, and 6 have wall-clock-overlapping exterior recordings. There is no simultaneous clean chest target. The table reports only: (1) change in sub-15-Hz motion energy relative to 15–200 Hz, where more negative is desirable; and (2) output/input attenuation in selected bands. These values cannot distinguish removed noise from removed pathology.

| Subject | Windows | Method | Motion/heart change (dB) | 15–40 Hz | 100–200 Hz | 200–400 Hz | 400–800 Hz |
|---|---:|---|---:|---:|---:|---:|---:|
| 2 | 201 | Bandpass | -7.563 | -1.773 | -0.335 | -0.315 | -0.141 |
| 2 | 201 | Ref. subtraction | -7.553 | -1.799 | -0.445 | -1.154 | -1.455 |
| 2 | 201 | Complex transfer | -7.827 | -1.714 | -0.409 | -0.396 | -0.220 |
| 2 | 201 | CardioSpecNet | -7.887 | -1.572 | -1.187 | -2.231 | -2.794 |
| 2 | 201 | Hybrid | -7.864 | -1.665 | -0.842 | -1.409 | -1.632 |
| 3 | 290 | Bandpass | -7.440 | -1.909 | -0.291 | -0.327 | -0.173 |
| 3 | 290 | Ref. subtraction | -6.689 | -2.500 | -3.288 | -5.265 | -3.745 |
| 3 | 290 | Complex transfer | -7.825 | -1.719 | -0.378 | -0.397 | -0.255 |
| 3 | 290 | CardioSpecNet | -6.986 | -1.971 | -3.467 | -5.652 | -4.613 |
| 3 | 290 | Hybrid | -7.302 | -1.936 | -2.039 | -3.138 | -2.632 |
| 5 | 219 | Bandpass | -7.348 | -1.162 | -0.306 | -0.127 | -0.049 |
| 5 | 219 | Ref. subtraction | -7.385 | -1.584 | -1.057 | -1.746 | -2.892 |
| 5 | 219 | Complex transfer | -7.600 | -1.203 | -0.374 | -0.203 | -0.135 |
| 5 | 219 | CardioSpecNet | -7.697 | -0.964 | -1.528 | -2.921 | -4.478 |
| 5 | 219 | Hybrid | -7.665 | -1.101 | -1.032 | -1.692 | -2.448 |
| 6 | 224 | Bandpass | -7.280 | -1.069 | -0.569 | -0.531 | -0.142 |
| 6 | 224 | Ref. subtraction | -7.459 | -1.610 | -2.944 | -4.756 | -3.104 |
| 6 | 224 | Complex transfer | -7.486 | -1.076 | -0.672 | -0.618 | -0.227 |
| 6 | 224 | CardioSpecNet | -7.412 | -0.764 | -2.405 | -4.663 | -4.247 |
| 6 | 224 | Hybrid | -7.465 | -0.929 | -1.678 | -2.769 | -2.402 |

Fusion consistently lies between the neural and complex-transfer outputs on real high-band attenuation. CardioSpecNet still attenuates 200–800 Hz by roughly 3–5 dB in several subjects; fusion reduces but does not eliminate that concern. Because those bands may contain murmurs, this remains a **deployment blocker** until pathology-stratified listening and downstream diagnostic evaluation are completed.

## Included checkpoint and compute profile

- Architecture: 534,134 trainable parameters; depth-3 2-D U-Net with two dual-axis bidirectional GRU blocks.
- Training: five CPU epochs, 1,024 fresh mixtures per epoch, 256 validation mixtures, batch size 16, SNR -10 to +20 dB, and 20% reference dropout. The selected checkpoint is zero-indexed epoch 3 (the fourth epoch).
- Selected training-time validation result: 9.106 dB mean SNR improvement, 5.510 dB mean SI-SDR improvement, and 0.739 mean correlation.
- Checkpoint size: approximately 6.28 MiB.
- Runtime profile in this container: at batch 1 and one PyTorch CPU thread, CardioSpecNet takes 25.47 ms per two-second window (real-time factor 0.0127) and the complete neural–adaptive hybrid takes 26.36 ms (0.0132), measured over 50 iterations after warm-up. Timing excludes file I/O, timestamp alignment, and overlap-add and is not a wearable-power measurement.

The implementation is offline/noncausal because it uses centered STFT frames, complete two-second windows, and bidirectional GRUs.

## Reproduction commands

```bash
python scripts/check_frequency_pipeline.py

python scripts/evaluate_frequency.py \
  --checkpoint checkpoints/frequency/cardiospecnet_demo/best.pt \
  --samples 512 \
  --reference_mode available \
  --target_mode bandlimited \
  --output_dir outputs/frequency_benchmark/reproduced_available

python scripts/evaluate_frequency.py \
  --checkpoint checkpoints/frequency/cardiospecnet_demo/best.pt \
  --samples 512 \
  --reference_mode available \
  --target_mode raw \
  --output_dir outputs/frequency_benchmark/reproduced_raw

python scripts/evaluate_frequency.py \
  --checkpoint checkpoints/frequency/cardiospecnet_demo/best.pt \
  --samples 512 \
  --reference_mode missing \
  --output_dir outputs/frequency_benchmark/reproduced_missing

python scripts/evaluate_real_proxy.py \
  --checkpoint checkpoints/frequency/cardiospecnet_demo/best.pt

python scripts/profile_frequency_runtime.py
```

## Limitations attached to every reported number

1. The old waveform CleanUNet checkpoint is absent, so its trained result cannot be reproduced or compared directly.
2. Synthetic evaluation uses the same augmentation family as training, although recordings, subjects, seeds, and mixtures are held out.
3. There is only one held-out clean participant and one held-out exterior-noise participant.
4. Fusion weights were calibrated on separate mixtures but the same held-out recordings; they need recalibration on a genuine development cohort.
5. Quiet recordings may contain residual physiological or sensor noise and are not a laboratory reference standard.
6. No pathology labels, murmur annotations, ECG timing, or clinician ratings are available for the real walking recordings.
7. Results summarized from papers in `RESEARCH_REVIEW.md` use different data, targets, contamination definitions, and metrics and are not numerically comparable.
