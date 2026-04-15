"""
Maithili TTS synthesis for Speech Understanding PA-2, Task 3.3.

Supports two TTS backends:
  Option A – Indic Parler-TTS  (ai4bharat/indic-parler-tts)
  Option B – Meta MMS TTS      (facebook/mms-tts-mai)

Includes Mel-Cepstral Distortion (MCD) computation for model comparison.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MIN_OUTPUT_SR = 22050


class MaithiliSynthesizer:
    """Synthesize Maithili speech with Indic-Parler-TTS, Meta MMS, or Coqui XTTS (voice cloning)."""

    def __init__(
        self,
        model_name: str = "xtts",
        speaker_embedding_path: Optional[str] = None,
        speaker_wav_path: Optional[str] = None,
        device: str = None,
    ):
        self.model_name = model_name
        if device is None:
            from scripts.device_utils import get_device
            device = get_device()
        if device == "mps":
            device = "cpu"
        self.device = device
        self.speaker_embedding = None
        self.speaker_wav_path = speaker_wav_path
        self._model = None
        self._processor = None
        self._sample_rate: int = MIN_OUTPUT_SR

        if speaker_embedding_path is not None:
            import torch

            self.speaker_embedding = torch.load(
                speaker_embedding_path, map_location=device, weights_only=True,
            )
            logger.info(
                "Loaded speaker embedding (%s) from %s",
                tuple(self.speaker_embedding.shape), speaker_embedding_path,
            )

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        if self.model_name == "parler":
            self._load_parler()
        elif self.model_name == "mms":
            self._load_mms()
        elif self.model_name == "xtts":
            self._load_xtts()
        else:
            raise ValueError(f"Unknown model: {self.model_name!r}. Use 'parler', 'mms', or 'xtts'.")

    def _load_parler(self):
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        model_id = "ai4bharat/indic-parler-tts"
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self._processor = AutoTokenizer.from_pretrained(model_id)

        if hasattr(self._model.config, "sampling_rate"):
            self._sample_rate = self._model.config.sampling_rate
        else:
            self._sample_rate = 22050
        logger.info("Loaded Indic Parler-TTS (sr=%d)", self._sample_rate)

    def _load_mms(self):
        from transformers import VitsModel, AutoTokenizer

        model_id = "facebook/mms-tts-mai"
        self._model = VitsModel.from_pretrained(model_id).to(self.device)
        self._processor = AutoTokenizer.from_pretrained(model_id)

        if hasattr(self._model.config, "sampling_rate"):
            self._sample_rate = self._model.config.sampling_rate
        else:
            self._sample_rate = 16000
        logger.info("Loaded Meta MMS-TTS (sr=%d)", self._sample_rate)

    def _load_xtts(self):
        """Load Coqui XTTS v2 for zero-shot voice cloning."""
        import torch

        # PyTorch 2.6+ defaults weights_only=True; XTTS checkpoints need these globals
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            torch.serialization.add_safe_globals([XttsConfig])
        except Exception:
            pass

        first_error = None
        try:
            from TTS.api import TTS
            self._tts_api = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            if self.device != "cpu":
                self._tts_api.to(self.device)
            self._sample_rate = 24000
            self._xtts_backend = "tts_api"
            logger.info("Loaded XTTS v2 via TTS API (sr=%d)", self._sample_rate)
            return
        except Exception as err1:
            first_error = err1
            logger.warning("TTS API failed (%s); trying direct XTTS load...", err1)

        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            from TTS.utils.manage import ModelManager

            manager = ModelManager()
            model_path, config_path, _ = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")

            config = XttsConfig()
            config.load_json(config_path)
            self._model = Xtts.init_from_config(config)
            self._model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=False)
            self._model.to(self.device)
            self._sample_rate = 24000
            self._xtts_backend = "direct"
            logger.info("Loaded XTTS v2 directly (sr=%d)", self._sample_rate)
            return
        except Exception as err2:
            logger.warning("Direct XTTS load also failed (%s)", err2)

        # Last resort: patch torch.load and retry
        try:
            logger.info("Retrying XTTS with torch.load weights_only=False...")
            _original_load = torch.load
            torch.load = lambda *a, **kw: _original_load(*a, **{**kw, "weights_only": False})
            from TTS.api import TTS
            self._tts_api = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            if self.device != "cpu":
                self._tts_api.to(self.device)
            self._sample_rate = 24000
            self._xtts_backend = "tts_api"
            torch.load = _original_load
            logger.info("Loaded XTTS v2 via patched torch.load (sr=%d)", self._sample_rate)
            return
        except Exception as err3:
            torch.load = _original_load
            raise RuntimeError(
                f"Could not load XTTS v2 after 3 attempts. "
                f"Error 1: {first_error}, Error 2: {err2}, Error 3: {err3}"
            )

    # ------------------------------------------------------------------
    # Segment-level synthesis
    # ------------------------------------------------------------------

    def synthesize_segment(
        self,
        text: str,
        speaker_embedding: Optional["torch.Tensor"] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize a single text segment.

        Returns:
            (audio_array, sample_rate)
        """
        import torch

        emb = speaker_embedding if speaker_embedding is not None else self.speaker_embedding

        if self.model_name == "parler":
            return self._synth_parler(text, emb)
        elif self.model_name == "xtts":
            return self._synth_xtts(text)
        else:
            return self._synth_mms(text)

    def _synth_parler(
        self, text: str, speaker_embedding: Optional["torch.Tensor"] = None,
    ) -> Tuple[np.ndarray, int]:
        import torch

        description = "A clear Maithili speaker with a natural tone"
        desc_inputs = self._processor(description, return_tensors="pt").to(self.device)
        text_inputs = self._processor(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            gen_kwargs = {
                "input_ids": desc_inputs.input_ids,
                "prompt_input_ids": text_inputs.input_ids,
            }
            output = self._model.generate(**gen_kwargs)

        audio = output.cpu().numpy().squeeze()
        return audio, self._sample_rate

    def _synth_mms(self, text: str) -> Tuple[np.ndarray, int]:
        import torch

        inputs = self._processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self._model(**inputs)

        audio = output.waveform.cpu().numpy().squeeze()
        return audio, self._sample_rate

    def _synth_xtts(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using XTTS v2 with zero-shot voice cloning from speaker_wav_path."""
        if self.speaker_wav_path is None or not Path(self.speaker_wav_path).exists():
            raise RuntimeError(
                "XTTS requires speaker_wav_path for voice cloning. "
                "Pass --speaker-wav pointing to your 60s voice recording."
            )

        backend = getattr(self, "_xtts_backend", "tts_api")

        if backend == "tts_api":
            wav = self._tts_api.tts(
                text=text,
                speaker_wav=self.speaker_wav_path,
                language="hi",
            )
            audio = np.array(wav, dtype=np.float32)
        else:
            import torch
            gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
                audio_path=[self.speaker_wav_path]
            )
            out = self._model.inference(
                text=text,
                language="hi",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
            )
            audio = np.array(out["wav"], dtype=np.float32)

        return audio, self._sample_rate

    # ------------------------------------------------------------------
    # Lecture-level synthesis
    # ------------------------------------------------------------------

    def synthesize_lecture(
        self,
        segments_json_path: str,
        output_path: str,
        pause_between_segments_sec: float = 0.5,
    ) -> str:
        """Synthesize a full lecture from a JSON of text segments.

        Expected JSON schema (list of dicts):
            [{"text": "...", "id": "seg_0"}, ...]
        or
            {"segments": [{"text": "..."}]}

        Concatenates segments with silence gaps.
        """
        import soundfile as sf

        segments = self._load_segments(segments_json_path)
        logger.info("Synthesising %d segments with model=%s", len(segments), self.model_name)

        all_audio: List[np.ndarray] = []
        sr = self._sample_rate

        pause_samples = int(pause_between_segments_sec * sr)
        silence = np.zeros(pause_samples, dtype=np.float32)

        for i, seg in enumerate(segments):
            text = seg if isinstance(seg, str) else seg.get("maithili_text", seg.get("text", seg.get("maithili", "")))
            if not text.strip():
                continue
            audio, seg_sr = self.synthesize_segment(text)
            if seg_sr != sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=seg_sr, target_sr=sr)
            all_audio.append(audio.astype(np.float32))
            if i < len(segments) - 1:
                all_audio.append(silence)
            if (i + 1) % 10 == 0:
                logger.info("  synthesised %d / %d segments", i + 1, len(segments))

        lecture_audio = np.concatenate(all_audio) if all_audio else np.zeros(sr, dtype=np.float32)

        # Upsample if below minimum output rate
        if sr < MIN_OUTPUT_SR:
            import librosa
            lecture_audio = librosa.resample(lecture_audio, orig_sr=sr, target_sr=MIN_OUTPUT_SR)
            sr = MIN_OUTPUT_SR

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), lecture_audio, sr)
        logger.info("Lecture audio (%d samples, sr=%d) saved to %s", len(lecture_audio), sr, out)
        return str(out)

    @staticmethod
    def _load_segments(path: str) -> list:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("segments", data.get("translations", []))
        return []

    # ------------------------------------------------------------------
    # MCD computation
    # ------------------------------------------------------------------

    def compute_mcd(
        self,
        ref_audio_path: str,
        synth_audio_path: str,
        sr: int = 22050,
        order: int = 24,
    ) -> float:
        """Compute Mel-Cepstral Distortion between reference and synthesis.

        MCD = (10 * sqrt(2) / ln(10)) * mean_over_frames(
            sqrt( sum_{d=1}^{D} (ref_c_d - synth_c_d)^2 )
        )
        Uses WORLD+pysptk for proper MCCs, falls back to librosa mel+DCT.
        DTW-aligns sequences before computing.
        """
        import librosa

        ref, _ = librosa.load(ref_audio_path, sr=sr, mono=True)
        syn, _ = librosa.load(synth_audio_path, sr=sr, mono=True)

        try:
            import pyworld as pw
            import pysptk

            alpha = pysptk.util.mcepalpha(sr)

            def _extract(audio):
                a64 = audio.astype(np.float64)
                f0, t = pw.harvest(a64, sr, frame_period=5.0)
                sp = pw.cheaptrick(a64, f0, t, sr)
                mcc = pysptk.sp2mc(sp, order=order, alpha=alpha)
                return mcc[:, 1:]

            ref_mcc = _extract(ref)
            syn_mcc = _extract(syn)
        except (ImportError, Exception):
            from scipy.fft import dct

            def _extract_fallback(audio):
                mel = librosa.feature.melspectrogram(
                    y=audio, sr=sr, n_fft=1024, hop_length=int(sr * 0.005),
                    n_mels=order * 2 + 1, power=1.0,
                )
                log_mel = np.log(mel + 1e-8)
                mcc = dct(log_mel, type=2, axis=0, norm="ortho")[1 : order + 1]
                return mcc.T

            ref_mcc = _extract_fallback(ref)
            syn_mcc = _extract_fallback(syn)

        from scripts.prosody_warping import dtw_align

        ref_flat = np.mean(ref_mcc, axis=1) if ref_mcc.shape[0] < ref_mcc.shape[1] else np.mean(ref_mcc, axis=0)
        syn_flat = np.mean(syn_mcc, axis=1) if syn_mcc.shape[0] < syn_mcc.shape[1] else np.mean(syn_mcc, axis=0)
        src_idx, tgt_idx = dtw_align(syn_flat, ref_flat)

        ref_aligned = ref_mcc[tgt_idx] if ref_mcc.shape[0] > ref_mcc.shape[1] else ref_mcc[:, tgt_idx].T
        syn_aligned = syn_mcc[src_idx] if syn_mcc.shape[0] > syn_mcc.shape[1] else syn_mcc[:, src_idx].T

        diff = ref_aligned - syn_aligned
        frame_dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        mcd = (10.0 * np.sqrt(2.0) / np.log(10.0)) * np.mean(frame_dist)

        logger.info("MCD(%s, %s) = %.3f dB", ref_audio_path, synth_audio_path, mcd)
        return float(mcd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Synthesize Maithili lecture audio (Task 3.3)",
    )
    parser.add_argument(
        "--text-json", type=str,
        default="outputs/maithili_translated_text.json",
        help="Path to JSON file with translated text segments",
    )
    parser.add_argument(
        "--speaker-embedding", type=str,
        default="outputs/audio/your_voice_embedding.pt",
        help="Path to speaker embedding .pt file",
    )
    parser.add_argument(
        "--output", type=str,
        default="outputs/audio/output_LRL_cloned.wav",
        help="Path for synthesised lecture output",
    )
    parser.add_argument(
        "--model", type=str,
        choices=["parler", "mms", "xtts"],
        default="xtts",
        help="TTS model backend (xtts for voice cloning)",
    )
    parser.add_argument(
        "--speaker-wav", type=str,
        default="outputs/audio/student_voice_ref.wav",
        help="Path to speaker reference WAV for XTTS voice cloning",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device",
    )
    parser.add_argument(
        "--pause-sec", type=float, default=0.5,
        help="Pause duration between segments (seconds)",
    )
    args = parser.parse_args()

    synth = MaithiliSynthesizer(
        model_name=args.model,
        speaker_embedding_path=args.speaker_embedding,
        speaker_wav_path=args.speaker_wav,
        device=args.device,
    )
    synth.synthesize_lecture(
        args.text_json,
        args.output,
        pause_between_segments_sec=args.pause_sec,
    )
    print(f"Done.  Output -> {args.output}")


if __name__ == "__main__":
    main()
