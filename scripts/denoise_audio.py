"""
Audio denoising and normalization for Speech Understanding PA-2.
Task 1.3: DeepFilterNet noise reduction with spectral subtraction fallback.

Provides two denoising strategies:
  1. DeepFilterNet — deep-learning-based real-time speech enhancement
  2. Spectral subtraction — classical signal processing fallback when
     DeepFilterNet is not available

All audio is normalized to the EBU R128 loudness target (-23 LUFS)
after denoising.
"""

import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate.

    Returns:
        (audio_array, sample_rate) tuple.
    """
    import torchaudio

    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    audio = waveform.squeeze(0).numpy()
    return audio, sr


def save_audio(audio: np.ndarray, sr: int, path: str) -> None:
    """Save audio array to file."""
    import torch
    import torchaudio

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio, np.ndarray):
        waveform = torch.from_numpy(audio).float()
    else:
        waveform = audio.float()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    torchaudio.save(str(out_path), waveform, sr)
    logger.info(f"Saved audio to {path} (sr={sr}, duration={waveform.shape[1]/sr:.2f}s)")


def normalize_audio(
    audio: np.ndarray, sr: int, target_lufs: float = -23.0
) -> np.ndarray:
    """Normalize audio loudness to target LUFS using pyloudnorm.

    Falls back to simple peak normalization if pyloudnorm is not available.

    Args:
        audio: 1-D float audio array.
        sr: Sample rate.
        target_lufs: Target integrated loudness in LUFS (EBU R128 default: -23).

    Returns:
        Loudness-normalized audio array.
    """
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        current_lufs = meter.integrated_loudness(audio)

        if np.isinf(current_lufs) or np.isnan(current_lufs):
            logger.warning("Loudness measurement failed (silent audio?); using peak normalization")
            return _peak_normalize(audio)

        normalized = pyln.normalize.loudness(audio, current_lufs, target_lufs)
        logger.info(f"Normalized loudness: {current_lufs:.1f} -> {target_lufs:.1f} LUFS")
        return normalized

    except ImportError:
        logger.warning("pyloudnorm not available; falling back to peak normalization")
        return _peak_normalize(audio)


def _peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Simple peak normalization."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio * (target_peak / peak)
    return audio


def denoise_deepfilternet(input_path: str, output_path: str) -> str:
    """Denoise audio using DeepFilterNet.

    DeepFilterNet is a deep-learning-based speech enhancement model
    optimized for real-time noise reduction. It must be installed locally
    (pip install deepfilternet).

    Args:
        input_path: Path to noisy audio file.
        output_path: Path to save denoised audio.

    Returns:
        Path to denoised audio file.
    """
    import torch

    audio, sr = load_audio(input_path, target_sr=48000)

    try:
        from df.enhance import enhance, init_df

        df_model, df_state, _ = init_df()

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        enhanced = enhance(df_model, df_state, audio_tensor)

        if isinstance(enhanced, torch.Tensor):
            enhanced_np = enhanced.squeeze().numpy()
        else:
            enhanced_np = np.array(enhanced).squeeze()

        enhanced_np = normalize_audio(enhanced_np, sr)
        save_audio(enhanced_np, sr, output_path)

        logger.info(f"DeepFilterNet denoising complete: {input_path} -> {output_path}")
        return output_path

    except ImportError:
        logger.error(
            "DeepFilterNet not installed. Install with: pip install deepfilternet. "
            "Falling back to spectral subtraction."
        )
        return denoise_spectral_subtraction(input_path, output_path)
    except Exception as e:
        logger.error(f"DeepFilterNet failed: {e}. Falling back to spectral subtraction.")
        return denoise_spectral_subtraction(input_path, output_path)


def denoise_spectral_subtraction(
    input_path: str,
    output_path: str,
    noise_frames: int = 10,
    oversubtraction: float = 2.0,
    spectral_floor: float = 0.01,
) -> str:
    """Denoise audio using spectral subtraction (classical DSP fallback).

    Estimates noise spectrum from the first N frames of audio (assumed to
    contain mostly noise / silence), then subtracts it from the full signal:

        S_clean(f) = max(|S_noisy(f)|^2 - alpha * |S_noise(f)|^2, floor)^0.5

    Args:
        input_path: Path to noisy audio file.
        output_path: Path to save denoised audio.
        noise_frames: Number of initial STFT frames to use for noise estimation.
        oversubtraction: Alpha factor for noise spectrum oversubtraction.
        spectral_floor: Minimum spectral magnitude to prevent musical noise.

    Returns:
        Path to denoised audio file.
    """
    from scipy import signal as scipy_signal

    audio, sr = load_audio(input_path, target_sr=16000)

    n_fft = 1024
    hop_length = 256
    window = "hann"

    freqs, times, stft_matrix = scipy_signal.stft(
        audio, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop_length,
    )

    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    noise_estimate = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    power_clean = magnitude ** 2 - oversubtraction * noise_estimate ** 2
    power_floor = (spectral_floor * noise_estimate) ** 2
    power_clean = np.maximum(power_clean, power_floor)

    magnitude_clean = np.sqrt(power_clean)

    stft_clean = magnitude_clean * np.exp(1j * phase)

    _, audio_clean = scipy_signal.istft(
        stft_clean, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop_length,
    )

    audio_clean = audio_clean[: len(audio)]
    audio_clean = normalize_audio(audio_clean, sr)

    save_audio(audio_clean, sr, output_path)
    logger.info(f"Spectral subtraction denoising complete: {input_path} -> {output_path}")
    return output_path


def denoise(
    input_path: str,
    output_path: str,
    method: str = "deepfilternet",
) -> str:
    """Dispatcher for audio denoising.

    Args:
        input_path: Path to noisy audio.
        output_path: Path for denoised output.
        method: Either "deepfilternet" or "spectral".

    Returns:
        Path to the denoised audio file.
    """
    input_p = Path(input_path)
    if not input_p.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    if method == "deepfilternet":
        return denoise_deepfilternet(input_path, output_path)
    elif method == "spectral":
        return denoise_spectral_subtraction(input_path, output_path)
    else:
        raise ValueError(f"Unknown denoising method: {method}. Use 'deepfilternet' or 'spectral'.")


def main():
    parser = argparse.ArgumentParser(
        description="Audio denoising and normalization (Task 1.3)"
    )
    parser.add_argument("--input", required=True, help="Path to input audio file")
    parser.add_argument("--output", required=True, help="Path to save denoised audio")
    parser.add_argument(
        "--method", choices=["deepfilternet", "spectral"], default="deepfilternet",
        help="Denoising method"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    denoise(args.input, args.output, method=args.method)
    logger.info("Denoising complete.")


if __name__ == "__main__":
    main()
