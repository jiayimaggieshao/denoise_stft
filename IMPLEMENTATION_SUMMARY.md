# Frequency-Domain Denoising Deliverable

This archive preserves the original waveform CleanUNet experiment and adds a complete, reproducible frequency-domain path for the two-microphone wearable prototype.

## Primary deliverables

- `src/frequency_model.py` — **CardioSpecNet**, a 534,134-parameter prior-anchored complex-STFT U-Net with compact recurrent modeling over both time and frequency.
- `src/frequency_data.py` — deterministic on-the-fly paired mixtures with independent chest/reference transfer paths, delay, equalization, transients, reference leakage/noise, and reference dropout.
- `src/frequency_baselines.py` — soft bandpass, reference spectral subtraction, per-frequency complex transfer cancellation, oracle reference, and calibrated reference-aware fusion.
- `src/frequency_loss.py` and `src/frequency_metrics.py` — phase-aware, scale-aware, and 15–800 Hz preservation objectives and metrics.
- `src/window_alignment.py` — wall-clock nearest-neighbor microphone alignment and exact sample-index overlap-add reconstruction.
- `train_frequency.py` — reproducible training/checkpoint loop.
- `scripts/evaluate_frequency.py` — paired benchmarks with explicit `bandlimited` and `raw` target modes.
- `scripts/evaluate_real_proxy.py` — transparent unpaired proxy analysis for real walking recordings.
- `scripts/run_frequency_inference.py` — timestamp-aligned real-recording inference and WAV/figure export.
- `scripts/profile_frequency_runtime.py` — CPU runtime profile for neural, classical, and complete hybrid paths.
- `scripts/summarize_frequency_benchmarks.py` — regenerates the compact benchmark CSV.
- `tests/test_frequency_pipeline.py` — shape/gradient, deterministic data, alignment, overlap-add, fusion, and missing-reference tests.

## Included trained and measured artifacts

- `checkpoints/frequency/cardiospecnet_demo/best.pt` — short five-epoch CPU engineering checkpoint.
- `outputs/frequency_benchmark/` — paired main, raw-target, missing-reference, SNR-stratified, runtime, and real-proxy results.
- `outputs/heart_w6_frequency_0-10s/` — ten-second real walking example with neural, adaptive, fused, reference, and bandpass WAV files.
- `analysis_outputs/` — spectral/coherence/reference-energy audit artifacts.

## Recommended commands

```bash
python scripts/check_frequency_pipeline.py

python scripts/evaluate_frequency.py \
  --checkpoint checkpoints/frequency/cardiospecnet_demo/best.pt \
  --samples 512 --reference_mode available --target_mode bandlimited

python scripts/run_frequency_inference.py \
  --input data/test_real/walking/step_0.1s/heart_w6_windows.npz \
  --checkpoint checkpoints/frequency/cardiospecnet_demo/best.pt \
  --start 0 --end 10 --export_baselines
```

The default real-recording output is the 55% CardioSpecNet / 45% complex-transfer monitoring fusion. It automatically falls back to CardioSpecNet for windows without a valid exterior-microphone timestamp match. Use `--fusion_neural_weight 0.20` for the calibrated raw-waveform-fidelity profile.

## Interpretation boundary

The synthetic benchmark supports engineering comparison but not clinical safety. The real walking data have no simultaneous clean target, pathology labels, or clinician ratings. The observed 200–800 Hz attenuation in several real proxy cases must be resolved with pathology-stratified evaluation before deployment.
