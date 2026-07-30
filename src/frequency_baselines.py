"""Classical and hybrid frequency-domain baselines for two-microphone PCG denoising."""
from __future__ import annotations

import torch
from torch import Tensor

from src.frequency_model import STFTConfig, soft_bandpass_response


def _prepare_pair(chest: Tensor, reference: Tensor | None) -> tuple[Tensor, Tensor]:
    if chest.ndim == 3 and chest.shape[1] == 1:
        chest = chest[:, 0]
    if reference is None:
        reference = torch.zeros_like(chest)
    elif reference.ndim == 3 and reference.shape[1] == 1:
        reference = reference[:, 0]
    if reference.shape != chest.shape:
        raise ValueError(f"reference shape {tuple(reference.shape)} != chest shape {tuple(chest.shape)}")
    return chest, reference


def _stft_ops(waveform: Tensor, config: STFTConfig) -> tuple[Tensor, Tensor, int]:
    window = torch.hann_window(config.win_length, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.stft(
        waveform,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=config.center,
        return_complex=True,
    )
    freqs = torch.fft.rfftfreq(config.n_fft, d=1.0 / config.sample_rate).to(waveform.device)
    bandpass = soft_bandpass_response(
        freqs,
        config.passband_low_hz,
        config.passband_high_hz,
        config.low_transition_hz,
        config.high_transition_hz,
    )[:, None]
    return spectrum, bandpass.to(spectrum.dtype), waveform.shape[-1]


def _istft(spectrum: Tensor, length: int, config: STFTConfig) -> Tensor:
    window = torch.hann_window(config.win_length, device=spectrum.device, dtype=spectrum.real.dtype)
    return torch.istft(
        spectrum,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=config.center,
        length=length,
    )


def bandpass_only(chest: Tensor, config: STFTConfig | None = None) -> Tensor:
    config = config or STFTConfig()
    chest, _ = _prepare_pair(chest, None)
    scale = chest.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
    normalized = chest / scale
    spectrum, bandpass, length = _stft_ops(normalized, config)
    filtered = spectrum * bandpass
    return _istft(filtered, length, config) * scale


def reference_spectral_subtraction(
    chest: Tensor,
    reference: Tensor,
    config: STFTConfig | None = None,
) -> Tensor:
    config = config or STFTConfig()
    chest, reference = _prepare_pair(chest, reference)
    scale = chest.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
    chest_norm = chest / scale
    reference_norm = reference / scale
    mixture, bandpass, length = _stft_ops(chest_norm, config)
    reference_spec, _, _ = _stft_ops(reference_norm, config)
    filtered_mixture = mixture * bandpass
    filtered_reference = reference_spec * bandpass

    eps = 1e-7
    y_mag = filtered_mixture.abs().clamp_min(eps)
    r_mag = filtered_reference.abs().clamp_min(eps)
    freqs = torch.fft.rfftfreq(config.n_fft, d=1.0 / config.sample_rate).to(chest.device)
    calibration = (freqs >= 850.0) & (freqs < 1_000.0)
    ratio = y_mag[:, calibration] / r_mag[:, calibration]
    transfer_scale = ratio.flatten(1).median(dim=1).values[:, None, None].clamp(0.05, 20.0)
    estimated_noise_power = (transfer_scale * r_mag).square()
    clean_power = (y_mag.square() - estimated_noise_power).clamp_min(0.0)
    gain = torch.sqrt(clean_power / y_mag.square().clamp_min(eps)).clamp(0.08, 1.0)
    enhanced = filtered_mixture * gain
    return _istft(enhanced, length, config) * scale


def complex_transfer_cancellation(
    chest: Tensor,
    reference: Tensor,
    config: STFTConfig | None = None,
) -> Tensor:
    """Estimate one complex transfer coefficient per frequency over each window."""
    config = config or STFTConfig()
    chest, reference = _prepare_pair(chest, reference)
    scale = chest.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
    chest_norm = chest / scale
    reference_norm = reference / scale
    mixture, bandpass, length = _stft_ops(chest_norm, config)
    reference_spec, _, _ = _stft_ops(reference_norm, config)
    filtered_mixture = mixture * bandpass
    filtered_reference = reference_spec * bandpass

    eps = 1e-7
    reference_power = filtered_reference.abs().square().mean(dim=-1, keepdim=True).clamp_min(eps)
    transfer = (filtered_mixture * filtered_reference.conj()).mean(dim=-1, keepdim=True) / reference_power
    reference_energy = reference_norm.square().mean(dim=-1)
    has_reference = (reference_energy > 1e-8).to(filtered_mixture.dtype)[:, None, None]
    cancelled = filtered_mixture - transfer * filtered_reference
    enhanced = has_reference * cancelled + (1.0 - has_reference) * filtered_mixture
    return _istft(enhanced, length, config) * scale


def hybrid_frequency_fusion(
    neural: Tensor,
    adaptive: Tensor,
    *,
    neural_weight: float,
    ref_available: Tensor | None = None,
) -> Tensor:
    if not 0.0 <= neural_weight <= 1.0:
        raise ValueError("neural_weight must be in [0, 1]")
    if neural.shape != adaptive.shape:
        raise ValueError(f"neural shape {tuple(neural.shape)} != adaptive shape {tuple(adaptive.shape)}")
    fused = neural_weight * neural + (1.0 - neural_weight) * adaptive
    if ref_available is None:
        return fused
    availability = ref_available.reshape(-1, *([1] * (neural.ndim - 1))).to(neural.dtype)
    return availability * fused + (1.0 - availability) * neural
