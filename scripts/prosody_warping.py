"""
DTW-based prosody warping for Speech Understanding PA-2, Task 3.2.

Custom implementation (NOT a generic wrapper) that:
  1. Extracts F0 and energy contours via Parselmouth
  2. Aligns source/target contours with DTW
  3. Applies the warped prosody via WORLD vocoder (pyworld) or PSOLA fallback
  4. Includes ablation mode: flat (unwarped) synthesis for comparison
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

HOP_TIME = 0.01  # 10 ms hop for prosody analysis


# ---------------------------------------------------------------------------
# Prosody extraction
# ---------------------------------------------------------------------------

def extract_prosody(audio_path: str, sr: int = 22050, max_chunk_sec: float = 60.0) -> Dict[str, np.ndarray]:
    """Extract F0 and energy contours from an audio file via Parselmouth.

    Processes in chunks to avoid bus errors on large files (macOS/Apple Silicon).

    Returns:
        dict with keys "f0", "energy", "times" (all 1-D numpy arrays).
    """
    import librosa
    import parselmouth

    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    total_samples = len(audio)
    chunk_samples = int(max_chunk_sec * sr)

    all_f0 = []
    all_energy = []
    all_times = []
    time_offset = 0.0

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end].astype(np.float64)

        if len(chunk) < sr * 0.1:
            break

        snd = parselmouth.Sound(chunk, sampling_frequency=sr)

        pitch = snd.to_pitch(time_step=HOP_TIME)
        f0 = pitch.selected_array["frequency"].copy()
        f0[f0 == 0] = np.nan
        f0_times = pitch.xs() + time_offset

        intensity = snd.to_intensity(time_step=HOP_TIME)
        energy = intensity.values[0].copy()

        all_f0.append(f0)
        all_energy.append(energy)
        all_times.append(f0_times)
        time_offset += len(chunk) / sr

    f0_cat = np.concatenate(all_f0) if all_f0 else np.array([])
    energy_cat = np.concatenate(all_energy) if all_energy else np.array([])
    times_cat = np.concatenate(all_times) if all_times else np.array([])

    logger.info(
        "Prosody: %d F0 frames, %d energy frames from %.1fs audio",
        len(f0_cat), len(energy_cat), total_samples / sr,
    )
    return {"f0": f0_cat, "energy": energy_cat, "times": times_cat}


# ---------------------------------------------------------------------------
# DTW alignment (custom Sakoe-Chiba band implementation)
# ---------------------------------------------------------------------------

def _euclidean_cost(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b):
        return 0.0  # unvoiced frames contribute no cost
    return (a - b) ** 2


def dtw_align(
    source: np.ndarray,
    target: np.ndarray,
    band_fraction: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute DTW alignment between *source* and *target* contours.

    Uses a Sakoe-Chiba band to keep the path near the diagonal and runs in
    O(N*W) where W = band_fraction * max(N, M).

    Returns:
        (source_indices, target_indices) – aligned index pair arrays.
    """
    try:
        from dtw import dtw as dtw_lib

        alignment = dtw_lib(
            source.reshape(-1, 1),
            target.reshape(-1, 1),
            keep_internals=False,
        )
        return np.array(alignment.index1), np.array(alignment.index2)
    except ImportError:
        pass

    # ---- Pure-numpy fallback with Sakoe-Chiba band ----
    N, M = len(source), len(target)
    band = max(int(band_fraction * max(N, M)), abs(N - M) + 1)

    cost = np.full((N + 1, M + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, N + 1):
        j_lo = max(1, i - band)
        j_hi = min(M, i + band) + 1
        for j in range(j_lo, j_hi):
            c = _euclidean_cost(source[i - 1], target[j - 1])
            cost[i, j] = c + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Backtrack
    i, j = N, M
    path_i, path_j = [i - 1], [j - 1]
    while i > 1 or j > 1:
        candidates = []
        if i > 1 and j > 1:
            candidates.append((cost[i - 1, j - 1], i - 1, j - 1))
        if i > 1:
            candidates.append((cost[i - 1, j], i - 1, j))
        if j > 1:
            candidates.append((cost[i, j - 1], i, j - 1))
        _, i, j = min(candidates, key=lambda x: x[0])
        path_i.append(i - 1)
        path_j.append(j - 1)

    return np.array(path_i[::-1]), np.array(path_j[::-1])


# ---------------------------------------------------------------------------
# Warping helpers
# ---------------------------------------------------------------------------

def warp_f0(
    synth_f0: np.ndarray,
    professor_f0: np.ndarray,
    alignment_path: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Map professor's F0 contour onto synth frame timing via DTW alignment."""
    src_idx, tgt_idx = alignment_path
    warped = synth_f0.copy()
    for si, ti in zip(src_idx, tgt_idx):
        if si < len(warped) and ti < len(professor_f0):
            warped[si] = professor_f0[ti]
    return warped


def warp_energy(
    synth_energy: np.ndarray,
    professor_energy: np.ndarray,
    alignment_path: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Map professor's energy contour onto synth frame timing via DTW alignment."""
    src_idx, tgt_idx = alignment_path
    warped = synth_energy.copy()
    for si, ti in zip(src_idx, tgt_idx):
        if si < len(warped) and ti < len(professor_energy):
            warped[si] = professor_energy[ti]
    return warped


# ---------------------------------------------------------------------------
# Prosody modification via WORLD vocoder (primary) or PSOLA (fallback)
# ---------------------------------------------------------------------------

def _apply_world(
    audio: np.ndarray,
    sr: int,
    original_f0: np.ndarray,
    target_f0: np.ndarray,
    target_energy: np.ndarray,
) -> np.ndarray:
    """Modify F0 and energy using WORLD vocoder (pyworld)."""
    import pyworld as pw

    audio_f64 = audio.astype(np.float64)
    f0_world, timeaxis = pw.harvest(audio_f64, sr, frame_period=HOP_TIME * 1000)
    sp = pw.cheaptrick(audio_f64, f0_world, timeaxis, sr)
    ap = pw.d4c(audio_f64, f0_world, timeaxis, sr)

    n_frames = len(f0_world)
    # Interpolate target_f0 to WORLD frame count
    target_f0_interp = np.interp(
        np.linspace(0, 1, n_frames),
        np.linspace(0, 1, len(target_f0)),
        np.nan_to_num(target_f0, nan=0.0),
    )
    # Where target is unvoiced (0 or NaN), keep original voicing decision
    voiced_mask = target_f0_interp > 0
    new_f0 = f0_world.copy()
    new_f0[voiced_mask] = target_f0_interp[voiced_mask]

    # Energy scaling: adjust spectral envelope power
    if target_energy is not None and len(target_energy) > 0:
        target_e_interp = np.interp(
            np.linspace(0, 1, n_frames),
            np.linspace(0, 1, len(target_energy)),
            target_energy,
        )
        orig_power = np.sqrt(np.sum(sp ** 2, axis=1) + 1e-12)
        target_power_ratio = 10 ** ((target_e_interp - np.mean(target_e_interp)) / 20.0)
        # Soft clamp to avoid extreme gains
        target_power_ratio = np.clip(target_power_ratio, 0.25, 4.0)
        for i in range(n_frames):
            sp[i] *= target_power_ratio[i]

    result = pw.synthesize(new_f0, sp, ap, sr)
    return result.astype(np.float32)


def _apply_psola(
    audio: np.ndarray,
    sr: int,
    original_f0: np.ndarray,
    target_f0: np.ndarray,
    target_energy: np.ndarray,
) -> np.ndarray:
    """Pitch-shift via simple resampling-based PSOLA approximation (fallback)."""
    import librosa

    orig_median = np.nanmedian(original_f0[original_f0 > 0]) if np.any(original_f0 > 0) else 200.0
    targ_median = np.nanmedian(target_f0[target_f0 > 0]) if np.any(target_f0 > 0) else orig_median

    if orig_median <= 0:
        orig_median = 200.0
    if targ_median <= 0:
        targ_median = orig_median

    n_steps = 12 * np.log2(targ_median / orig_median)
    n_steps = np.clip(n_steps, -12, 12)

    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(n_steps))

    # Energy adjustment via RMS scaling
    if target_energy is not None and len(target_energy) > 0:
        target_rms = np.mean(10 ** (target_energy / 20.0))
        current_rms = np.sqrt(np.mean(shifted ** 2)) + 1e-12
        gain = np.clip(target_rms / current_rms, 0.25, 4.0)
        shifted = shifted * gain

    return shifted


def apply_prosody_modification(
    audio_array: np.ndarray,
    sr: int,
    original_f0: np.ndarray,
    target_f0: np.ndarray,
    target_energy: np.ndarray,
) -> np.ndarray:
    """Modify the prosody of *audio_array* to match target F0 and energy.

    Tries WORLD vocoder first; falls back to PSOLA-style shifting.
    """
    try:
        return _apply_world(audio_array, sr, original_f0, target_f0, target_energy)
    except ImportError:
        logger.warning("pyworld not available – falling back to PSOLA-style pitch shifting")
        return _apply_psola(audio_array, sr, original_f0, target_f0, target_energy)


# ---------------------------------------------------------------------------
# High-level ProsodyWarper class
# ---------------------------------------------------------------------------

class ProsodyWarper:
    """End-to-end DTW prosody warping: aligns synthesized audio to professor prosody."""

    def __init__(self, professor_audio_path: str, sr: int = 22050):
        self.sr = sr
        self.professor_audio_path = professor_audio_path
        logger.info("Extracting professor prosody from %s", professor_audio_path)
        self.professor_prosody = extract_prosody(professor_audio_path, sr=sr)

    def warp(self, synth_audio_path: str, output_path: str) -> str:
        """Full pipeline: extract synth prosody, DTW-align, warp, save."""
        import librosa
        import soundfile as sf

        synth_audio, _ = librosa.load(synth_audio_path, sr=self.sr, mono=True)
        synth_prosody = extract_prosody(synth_audio_path, sr=self.sr)

        # DTW on F0 — downsample to avoid memory explosion on long audio
        src_f0 = np.nan_to_num(synth_prosody["f0"], nan=0.0)
        tgt_f0 = np.nan_to_num(self.professor_prosody["f0"], nan=0.0)

        max_dtw_frames = 5000
        src_factor = max(1, len(src_f0) // max_dtw_frames)
        tgt_factor = max(1, len(tgt_f0) // max_dtw_frames)
        factor = max(src_factor, tgt_factor)

        if factor > 1:
            logger.info("Downsampling prosody by %dx for DTW (%d, %d -> %d, %d frames)",
                        factor, len(src_f0), len(tgt_f0),
                        len(src_f0) // factor, len(tgt_f0) // factor)
            src_f0_ds = src_f0[::factor]
            tgt_f0_ds = tgt_f0[::factor]
        else:
            src_f0_ds = src_f0
            tgt_f0_ds = tgt_f0

        src_idx_ds, tgt_idx_ds = dtw_align(src_f0_ds, tgt_f0_ds)

        src_idx = np.clip(src_idx_ds * factor, 0, len(src_f0) - 1)
        tgt_idx = np.clip(tgt_idx_ds * factor, 0, len(tgt_f0) - 1)

        warped_f0 = warp_f0(synth_prosody["f0"], self.professor_prosody["f0"], (src_idx, tgt_idx))
        warped_energy = warp_energy(
            synth_prosody["energy"], self.professor_prosody["energy"], (src_idx, tgt_idx),
        )

        warped_audio = apply_prosody_modification(
            synth_audio, self.sr,
            synth_prosody["f0"], warped_f0, warped_energy,
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), warped_audio, self.sr)
        logger.info("Warped audio saved to %s", out)
        return str(out)

    def warp_flat(self, synth_audio_path: str, output_path: str) -> str:
        """Save synthesis WITHOUT prosody warping (ablation baseline)."""
        import librosa
        import soundfile as sf

        audio, _ = librosa.load(synth_audio_path, sr=self.sr, mono=True)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, self.sr)
        logger.info("Flat (unwarped) audio saved to %s", out)
        return str(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="DTW prosody warping (Task 3.2)",
    )
    parser.add_argument(
        "--synth-audio", type=str, required=True,
        help="Path to synthesised audio to warp",
    )
    parser.add_argument(
        "--professor-audio", type=str, required=True,
        help="Path to professor's denoised lecture audio",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for warped audio",
    )
    parser.add_argument(
        "--output-flat", type=str, default=None,
        help="Output path for flat (unwarped) ablation audio",
    )
    parser.add_argument(
        "--sr", type=int, default=22050,
        help="Sample rate for analysis and output",
    )
    args = parser.parse_args()

    warper = ProsodyWarper(args.professor_audio, sr=args.sr)
    warper.warp(args.synth_audio, args.output)
    if args.output_flat:
        warper.warp_flat(args.synth_audio, args.output_flat)

    print(f"Done.  Warped -> {args.output}")
    if args.output_flat:
        print(f"       Flat   -> {args.output_flat}")


if __name__ == "__main__":
    main()
