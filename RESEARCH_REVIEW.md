# Research Review and Design Rationale

Review date: 2026-07-23

## Executive conclusion

The literature does not support a single universally best PCG denoiser. Recent high-performing work converges on four ideas: operate on time-frequency structure; use temporal context; preserve phase or predict complex spectra rather than masking magnitude alone; and evaluate whether clinically relevant features survive, not only whether waveform SNR improves. Wearable systems also increasingly include a second air/background microphone, but adaptive cancellation remains sensitive to transfer-path and synchronization assumptions.

For this repository, the best-supported engineering choice is therefore a **small, bounded complex-STFT network conditioned on exterior-microphone magnitude**, trained with aggressive transfer-path randomization and paired with strong classical two-channel cancellation. A full speech-sized separation network would be poorly matched to the dataset size and wearable target. A purely adaptive filter is too dependent on a stable transfer path to be the only method. The implementation is named **CardioSpecNet**; its calibrated convex fusion with complex transfer cancellation is the default reference-available monitoring output, and every component remains a first-class inference export.

## Most relevant evidence

### Lightweight spectrogram U-Nets for PCG denoising

Duggan, Temko, Sarana, Factor, and Popovici, “Denoising of Heart Sounds Using Lightweight FCNs and Spectrograms With and Without Context,” *IEEE Access*, vol. 13, pp. 77656–77672, 2025, DOI `10.1109/ACCESS.2025.3566288`.

This is the closest recent frequency-domain study to the supplied task. It uses 4 kHz PCG, a 256-point FFT, a 128-sample analysis window, a 32-sample hop, and lightweight Spleeter-style U-Nets. Its best context model reports an overall average denoising improvement of 10.322 dB across its synthetic contamination tests and was demonstrated on an edge device. The important caveat is that its data, noise construction, and metric definitions differ from this repository, so the number is not a benchmark target. CardioSpecNet retains the small 2-D U-Net/context idea but predicts a bounded complex mask and explicitly measures 200–800 Hz preservation.

### Complex time-frequency full/sub-band modeling

Wang et al., “TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural Speaker Separation,” *ICASSP*, 2023, arXiv:`2209.03952`; and “TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation,” *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2023, arXiv:`2211.12433`.

TF-GridNet alternates intra-frame spectral modeling, sub-band temporal modeling, and full-band attention while predicting complex time-frequency targets. It reports 23.4–23.5 dB SI-SDR improvement on WSJ0-2mix, but that is a speech-separation result and is not numerically transferable to PCG denoising. The architectural lesson is still valuable: alternate modeling over time and frequency instead of treating a spectrogram as an ordinary image. CardioSpecNet uses a compact adaptation—two bidirectional GRU passes over time and frequency inside a 534k-parameter U-Net—rather than a multi-million-parameter speech model.

### Multitask denoising that explicitly preserves valvular-disease features

Lee and Shin, “Multitask learning-based phonocardiogram denoising model for preserving valvular heart disease characteristics,” *Physiological Measurement*, July 2026, DOI `10.1088/1361-6579/ae8caa`.

This is the newest directly relevant paper found in the review. It jointly trains spectrogram denoising and valvular-heart-disease classification. The publisher abstract reports an absolute mean SI-SDR of 14.05 dB at -5 dB input SNR, improvement up to 9.69 dB over prior work for hospital/lung noise, and classification accuracy above 98.5% across most 5 and 10 dB conditions. Those values use a different dataset and nested cross-validation and are not comparable to this repository. The central lesson is stronger than the headline scores: pathology preservation should be an explicit supervised task, not inferred from waveform cleanliness.

### Self-supervised PCG denoising and pathology preservation

Abraham et al., “SSC-UNet: UNet with Self-Supervised Contrastive Learning for Phonocardiography Noise Reduction,” arXiv:`2601.10735`, 2026 preprint.

SSC-UNet combines Noise2Noise-style self-supervision, recurrent skip processing, augmentation, and contrastive learning. The paper reports 12.98 dB filtered SNR under 10 dB hospital noise and downstream classification sensitivity improving from 27% to 88%. This supports a later self-supervised adaptation stage on real walking recordings, where no clean target is available. It does not remove the need for pathology-stratified validation, and its reported numbers are not directly comparable with the present split.

### Real-time denoising under unseen noise

Ali et al., “An End-to-End Deep Learning Framework for Real-Time Denoising of Heart Sounds for Cardiac Disease Detection in Unseen Noise,” *IEEE Access*, vol. 11, 2023, DOI `10.1109/ACCESS.2023.3292551`.

The proposed LU-Net is important less as an architecture template than as an evaluation template: denoising is assessed together with downstream cardiac-disease detection and unseen noise. That motivates held-out exterior-noise recordings, reference dropout, and the requirement to score a frozen murmur/disease model after denoising. The supplied archive has no pathology labels or validated classifier, so a genuine task-aware result cannot be manufactured here.

### Clinically meaningful, feature-aware denoising

Bauxell et al., “Clinically Meaningful Phonocardiogram Denoising with Feature Aware Fine-Tuning for CHD Detection,” *IEEE Sensors Journal*, 2026, DOI `10.1109/JSEN.2026.3696544`.

This work explicitly fine-tunes denoising around downstream congenital-heart-disease features. It reinforces the key safety point: the numerically cleanest output need not be the diagnostically safest output. CardioSpecNet includes band-energy and over-attenuation penalties as proxies, but true feature-aware tuning requires annotations or a validated frozen diagnostic network.

### Two-microphone wearable evidence

Rong et al., “Wearable Electro-Phonocardiography Device for Cardiovascular Disease Monitoring,” *IEEE Statistical Signal Processing Workshop*, 2023.

This wearable uses a chest-facing heart microphone and a rear-facing background-noise microphone—the same physical concept as the supplied prototype—and evaluates NLMS and iterative Wiener cancellation. It confirms that the second microphone is a useful design choice, while also motivating a strong adaptive-filter comparator rather than assuming a neural model is automatically superior.

Li et al., “The Patchkeeper: An Integrated Wearable Electronic Stethoscope with Multiple Sensors,” arXiv:`2407.11837`, 2024.

Patchkeeper similarly couples one microphone to the stethoscope head and a second microphone to air for environmental tracking, noise cancellation, and privacy protection. This supports retaining synchronized stereo acquisition and improving timestamp/sample-clock metadata at the hardware level.

Herath et al., “A Simultaneous ECG-PCG Acquisition System with Real-Time Burst-Adaptive Noise Cancellation,” arXiv:`2510.23819`, 2025/2026 preprint.

This work uses burst-adaptive NLMS for low-complexity embedded cancellation and reports large SNR gains on its own evaluation. Because it is a preprint with different acquisition and SNR definitions, its headline number should not be compared with this repository. It nevertheless strengthens the case for keeping a causal adaptive baseline in the future embedded implementation.

### Quality assessment and broad-domain pretraining

Despotovic, Pocta, and Zgank, “CardioPHON: Quality Assessment and Self-Supervised Pretraining for Screening of Cardiac Function Based on Phonocardiogram Recordings,” arXiv:`2511.04533`, 2025 preprint.

CardioPHON combines a recording-quality gate with self-supervised pretraining on six public heart-sound datasets. For deployment, a quality/reject option may be safer than forcing a denoised answer for every window. Its released representation is also a plausible future source of a frozen pathology/quality loss, subject to licensing and independent validation.

## Architecture options considered

| Option | Strength | Reason not selected as the sole primary path |
|---|---|---|
| Magnitude-only 2-D U-Net | Proven, small, easy to deploy | Reuses noisy phase and underuses the exterior reference |
| Full TF-GridNet/CrossNet | Excellent complex T-F modeling | Excess capacity and training demand for this small PCG corpus and wearable target |
| Speech models such as DeepFilterNet | Efficient and phase-aware | Speech harmonic/noise priors and perceptual objectives are not automatically safe for S1/S2 and murmurs |
| Waveform LU-Net/CleanUNet | End-to-end and potentially causal | The original repo result is marginal; the reference channel and clear spectral structure are unused |
| NLMS/Wiener/complex transfer cancellation | Interpretable, low compute, naturally two-channel | Requires a sufficiently stable transfer relationship; clothing/contact motion can be nonstationary and phase coherence is weak in these files |
| **Prior-anchored complex U-Net with compact T/F GRUs** | Uses reference magnitude, corrects phase, bounded behavior, modest size | Selected learned component; still requires clinical validation |
| **Calibrated neural–adaptive fusion** | Combines complementary neural band preservation and adaptive waveform fidelity | Selected reference-available monitoring output; calibration is repository-specific |

## Why the implemented STFT setup differs from speech defaults

At 4 kHz, a 256-sample Hann window spans 64 ms and a 64-sample hop spans 16 ms. A 512-point FFT yields 7.8125 Hz bin spacing without increasing the physical window. This is a compromise: it has enough frequency resolution to separate strong sub-15-Hz walking drift from the main heart-sound bands while retaining substantially better transient resolution than the legacy 256–1,024 ms loss windows.

The model applies a smooth transition from 5 to 15 Hz and another from 800 to 1,000 Hz. The 15–800 Hz range is treated as a **coverage and preservation objective**, not proof that every output retains every murmur. The real walking proxy results show that this distinction matters.

## Generalization strategy implemented

1. **Recording-level separation:** clean subject 4 and exterior-noise subject 6 are excluded from training and used for deterministic validation/benchmarking.
2. **Overlap control:** the 0.1-second archive has 95% overlap, so source pools are thinned by ten before random mixing.
3. **Transfer-path randomization:** chest and exterior paths differ in delay, short impulse response, smooth frequency response, gain, polarity, independent contamination, and transient rubbing.
4. **Reference imperfections:** training injects heart leakage, sensor noise, and 20% complete reference dropout.
5. **Conservative output constraints:** the learned magnitude mask is capped at 1.2 and phase correction at ±0.35 rad; initialization reproduces the analytical reference-subtraction prior.
6. **Metric diversity:** evaluation reports scale-dependent SNR, SI-SDR, correlation, log-spectral distance, and five band-energy errors, plus unpaired real-data proxies.
7. **Classical challenge baseline:** per-frequency complex transfer cancellation is evaluated and exported alongside the model.
8. **Calibrated, reference-aware fusion:** a 55% neural monitoring profile and a 20% neural raw-fidelity profile are selected on a separate deterministic calibration seed, then evaluated on a fresh seed. Missing exterior windows fall back to the dropout-trained neural estimate. The same held-out source recordings are reused, so external calibration is still required.

## Generalization that remains untested

The supplied data cannot establish cross-device, cross-site, posture, clothing, skin-contact, age, sex, body-habitus, or pathology-stratified generalization. It also cannot validate patient speech removal separately from internal lung/organ sounds because those components are not labeled. The next decisive study is:

- collect simultaneous chest, exterior, and ECG/annotation data under controlled artifact types;
- split strictly by participant and recording session;
- include normal and pathological murmurs across devices/sites;
- evaluate raw, bandpass, adaptive cancellation, CardioSpecNet, calibrated fusion, and a self-supervised adaptation variant;
- score signal metrics, blinded clinician ratings, event timing, and a frozen diagnostic classifier;
- include an abstention/quality gate and report failure rates, not only mean scores.

## Reproducibility caveat

The included checkpoint is a short CPU-trained engineering checkpoint, not a claim of literature-level convergence. Paper results above use different datasets, contamination definitions, target construction, and metrics. They are design evidence, not directly comparable leaderboard numbers.
