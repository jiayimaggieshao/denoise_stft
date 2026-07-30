# CardioSpecNet Frequency-Domain Pipeline

## Goal

Denoise 4 kHz, two-second chest-microphone windows while using the exterior microphone as a potentially non-phase-coherent noise reference. The design is conservative: it removes obvious motion/environmental contamination, constrains learned amplification, and keeps representation and loss coverage through 800 Hz. Preservation of pathological murmur content is a validation target, not a claim established by the supplied data.

## Signal path

1. **Input conversion** — int16 windows are converted to float in `[-1, 1]`. No independent external per-window normalization is applied.
2. **Internal scaling** — both channels are divided by the chest-window RMS; the output is multiplied by the same scale.
3. **STFT** — 512-point FFT, 256-sample Hann window (64 ms), 64-sample hop (16 ms), centered offline analysis. The 512-point transform gives a 7.8125 Hz frequency grid while retaining a 64 ms physical window.
4. **Soft bandpass** — raised-cosine transition from 5 to 15 Hz and from 800 to 1,000 Hz. This suppresses severe walking drift while retaining model coverage through the commonly used PCG analysis band.
5. **Reference prior** — a robust chest/reference transfer scale is estimated in 850–1,000 Hz, above the protected band. Reference power is subtracted to form a conservative magnitude-gain prior with a floor of 0.08. Missing references produce an identity prior.
6. **Learned features** — compressed chest real and imaginary STFT, chest log magnitude, reference log magnitude, log magnitude ratio, per-frequency log-energy similarity, analytical prior gain, and reference-availability flag.
7. **Network** — a depth-3 lightweight 2-D U-Net. At the bottleneck, two dual-axis recurrent blocks alternate bidirectional GRUs over time and frequency, a compact adaptation of the full/sub-band modeling idea used by TF-GridNet.
8. **Output mask** — the network refines the analytical prior with a bounded magnitude mask (maximum 1.2×) and a bounded phase correction (±0.35 rad). Zero-initialized output weights make an untrained model equal the conservative prior rather than an arbitrary random denoiser.
9. **ISTFT** — enhanced complex spectrum is converted back to an 8,000-sample waveform and rescaled.
10. **Recording reconstruction** — window outputs are Hann overlap-added using stored `start_idx`; exterior windows are selected by nearest `start_wall_epoch_us`, subject to a configurable tolerance.

11. **Reference-aware neural–adaptive fusion** — the deployable monitoring output is a convex combination of CardioSpecNet and complex transfer cancellation. The default neural weight is 0.55; the raw-waveform-fidelity profile uses 0.20. When an exterior window is missing or outside the timestamp tolerance, fusion automatically returns the standalone neural estimate for that window rather than blending it with an uninformative adaptive branch.

Default neural model: 534,134 trainable parameters (`base_channels=12`, two time/frequency grid blocks). The included checkpoint uses this configuration.

## Why not rely only on direct adaptive subtraction?

The paired walking analysis shows low waveform coherence even when the recordings overlap in wall time. Independent sample clocks, acoustic propagation, clothing/contact paths, and dropped samples make sample-level cancellation unreliable. Magnitude/energy tracks contamination better, so CardioSpecNet conditions its mask on the exterior channel without importing exterior phase.

The repository nevertheless includes a strong frequency-domain comparator, `complex_transfer_cancellation()`. It estimates one complex exterior-to-chest transfer coefficient per frequency over each complete two-second window. The adaptive baseline leads the standalone neural model on SI-SDR and correlation, while CardioSpecNet leads on scale-dependent SNR and synthetic 100–800 Hz band-energy fidelity. Their calibrated 55%/45% fusion improves the held-out aggregate trade-off: 9.287 dB SNR improvement, 6.026 dB SI-SDR improvement, 0.692 correlation, and 8.909 dB 15–800 Hz log-spectral distance. All components are exported during inference because the real recordings do not contain a clean target that would justify declaring one universally safer.

## Output profiles and calibration

`evaluate_frequency.py` exposes two explicit objectives:

- `--target_mode bandlimited` evaluates against the soft-bandpassed quiet target used for training and defaults to a hybrid neural weight of 0.55.
- `--target_mode raw` evaluates against the original unfiltered quiet waveform and defaults to a hybrid neural weight of 0.20.

The weights were selected on 256 deterministic calibration mixtures (seed 5151) and evaluated on a separate 512-mixture seed (42424). Both sets use the same held-out source recordings, so the weights are repository-specific calibration values rather than physiological constants. `run_frequency_inference.py` defaults to the monitoring hybrid, automatically uses CardioSpecNet on unmatched windows, and accepts `--fusion_neural_weight 0.20` for the raw-fidelity profile.

## Synthetic training distribution

`SyntheticFrequencyDataset` samples a clean sitting window and one or two exterior-noise windows. It then applies:

- independent random chest and reference delays up to ±50 ms;
- short random impulse responses and smooth random equalization;
- random gain and polarity;
- 0–35% chest-only independent noise;
- optional clothing-like transients;
- 0–8% heart leakage into the exterior reference;
- reference sensor noise;
- 20% reference dropout;
- a uniform target SNR from −10 to +20 dB, plus 8% clean identity cases.

Noise is scaled by its energy after the same broad PCG bandpass, but the raw low-frequency contamination is retained in the chest input. This prevents sub-15-Hz motion from being underestimated.

## Training loss

The objective combines:

- target-normalized waveform L1;
- target-normalized, frequency-weighted complex-STFT L1;
- log-magnitude L1;
- normalized first-difference L1;
- direct scale-dependent SNR loss;
- waveform correlation loss;
- symmetric log-energy error in five clinical bands;
- asymmetric missing-energy penalty from 15 to 800 Hz.

Frequency weights are largest from 15–200 Hz and remain substantial from 200–800 Hz. This is deliberately different from speech denoisers that may remove non-harmonic high-frequency structure.

## Commands

### Verify the implementation

```bash
PYTHONPATH=. python scripts/check_frequency_pipeline.py
```

### Full training

```bash
PYTHONPATH=. python train_frequency.py \
  --step 0.1s \
  --source_stride 10 \
  --epochs 60 \
  --samples_per_epoch 20000 \
  --val_samples 2000 \
  --batch_size 16 \
  --base_channels 12 \
  --grid_blocks 2
```

The default source stride intentionally reduces the 95%-overlap archive to an effective one-second source stride. Mixing remains newly randomized each epoch.

### Reproduce a fast end-to-end smoke run

```bash
PYTHONPATH=. python train_frequency.py --smoke --device cpu
```

### Paired synthetic benchmark

```bash
PYTHONPATH=. python scripts/evaluate_frequency.py \
  --checkpoint checkpoints/frequency/cardiospecnet_demo/best.pt \
  --samples 1000 \
  --reference_mode available \
  --target_mode bandlimited
```

Repeat with `--target_mode raw` for exact quiet-waveform fidelity or `--reference_mode missing` to measure single-channel fallback.

### Real walking inference

```bash
PYTHONPATH=. python scripts/run_frequency_inference.py \
  --input data/test_real/walking/step_0.1s/heart_w6_windows.npz \
  --checkpoint checkpoints/frequency/cardiospecnet_demo/best.pt \
  --start 0 --end 20 \
  --primary_method hybrid_fusion \
  --fusion_neural_weight 0.55 \
  --export_baselines
```

For subjects 2, 3, 5, and 6, the script auto-discovers the corresponding exterior-noise archive. Supply `--reference` explicitly for another synchronized recording.

### Real-data proxy analysis

```bash
PYTHONPATH=. python scripts/evaluate_real_proxy.py \
  --checkpoint checkpoints/frequency/cardiospecnet_demo/best.pt
```

### CPU runtime profile

```bash
PYTHONPATH=. python scripts/profile_frequency_runtime.py
```

## Deployment limitations

The current model is **offline**, not a final streaming wearable implementation: centered STFT analysis and bidirectional recurrent layers use future context inside each two-second window. A causal deployment should switch to left-context STFT framing, unidirectional temporal recurrence, state caching, and measured end-to-end latency.

More importantly, CardioSpecNet attenuated 200–800 Hz by roughly 3–5 dB in several real walking proxy tests. The 55% hybrid reduces this attenuation to an intermediate level but does not resolve the uncertainty. Those bands may contain murmurs, and there is no simultaneous clean target or pathology annotation with which to distinguish noise removal from diagnostic signal removal. This is a deployment blocker. Before clinical use, evaluate blinded listening, murmur/event annotations, and a frozen downstream diagnostic model on external devices and sites; retain the complex-transfer and bandpass outputs as safety comparators during that study.
