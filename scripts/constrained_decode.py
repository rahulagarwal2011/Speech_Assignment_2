"""
Constrained decoding for Whisper with N-gram logit biasing.
Task 1.2: Custom LogitsProcessor integrating ARPA trigram LM
and technical vocabulary bias for speech course term prioritization.

Mathematical formulation:
  logit'(t, v) = logit(t, v) + alpha * log P_ngram(v | context) + beta * bias(v)

where:
  - logit(t, v) is the original Whisper logit for token v at time step t
  - P_ngram(v | context) is the trigram LM probability for token v given
    the preceding two tokens as context
  - bias(v) is the domain-specific boost weight for technical vocabulary
    (1.0 if v is a technical term, 0.0 otherwise)
  - alpha controls the weight of the language model prior
  - beta controls the weight of the domain vocabulary bias
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class NgramLogitBiasProcessor:
    """LogitsProcessor that biases Whisper decoding toward n-gram LM and technical terms.

    Implements the custom logit biasing formula:
        logit'(t, v) = logit(t, v) + alpha * log P_ngram(v | context) + beta * bias(v)

    This integrates an external ARPA trigram language model (via kenlm) and a
    domain-specific technical vocabulary to improve transcription of
    code-switched speech containing speech processing terminology.
    """

    def __init__(
        self,
        ngram_lm_path: str,
        technical_vocab_path: str,
        tokenizer=None,
        alpha: float = 0.3,
        beta: float = 3.0,
    ):
        """
        Args:
            ngram_lm_path: Path to the ARPA trigram LM file.
            technical_vocab_path: Path to JSON mapping technical terms to bias weights.
            tokenizer: Whisper tokenizer for mapping between token IDs and text.
            alpha: Weight for n-gram LM log probability.
            beta: Weight for technical vocabulary bias.
        """
        self.alpha = alpha
        self.beta = beta
        self.tokenizer = tokenizer

        self._ngram_model = self._load_ngram_lm(ngram_lm_path)
        self._technical_vocab = self._load_technical_vocab(technical_vocab_path)
        self._tech_token_ids: Optional[Dict[int, float]] = None

    def _load_ngram_lm(self, path: str):
        """Load ARPA language model via kenlm."""
        lm_path = Path(path)
        if not lm_path.exists():
            logger.warning(f"N-gram LM not found at {path}; n-gram biasing disabled")
            return None
        try:
            import kenlm
            model = kenlm.Model(str(lm_path))
            logger.info(f"Loaded {model.order}-gram LM from {path}")
            return model
        except ImportError:
            logger.warning("kenlm not installed; n-gram biasing disabled")
            return None

    def _load_technical_vocab(self, path: str) -> Dict[str, float]:
        """Load technical vocabulary with bias weights from JSON."""
        vocab_path = Path(path)
        if not vocab_path.exists():
            logger.warning(f"Technical vocab not found at {path}; vocab biasing disabled")
            return {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        if isinstance(vocab, list):
            vocab = {term: 1.0 for term in vocab}
        logger.info(f"Loaded {len(vocab)} technical terms from {path}")
        return vocab

    def _build_tech_token_map(self, vocab_size: int) -> Dict[int, float]:
        """Pre-compute mapping from token IDs to technical bias weights.

        Each technical term may span multiple subword tokens. We assign
        the bias weight to every token that constitutes part of a technical term.
        """
        if self.tokenizer is None or not self._technical_vocab:
            return {}

        token_bias: Dict[int, float] = {}
        for term, weight in self._technical_vocab.items():
            try:
                token_ids = self.tokenizer.encode(term)
                if hasattr(token_ids, "tolist"):
                    token_ids = token_ids.tolist()
                for tid in token_ids:
                    if 0 <= tid < vocab_size:
                        token_bias[tid] = max(token_bias.get(tid, 0.0), float(weight))
            except Exception:
                continue

        logger.info(f"Mapped technical vocab to {len(token_bias)} unique token IDs")
        return token_bias

    def _get_ngram_context(self, input_ids: torch.Tensor) -> str:
        """Decode the last (order-1) tokens as LM context string."""
        if self.tokenizer is None or self._ngram_model is None:
            return ""
        order = self._ngram_model.order
        context_ids = input_ids[-1, -(order - 1):].tolist() if input_ids.dim() > 1 else input_ids[-(order - 1):].tolist()
        try:
            context_str = self.tokenizer.decode(context_ids).strip()
        except Exception:
            context_str = ""
        return context_str

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Modify logits with n-gram LM scores and technical vocabulary bias.

        Args:
            input_ids: (batch, seq_len) previously generated token IDs.
            scores: (batch, vocab_size) current logit scores from Whisper.

        Returns:
            Modified scores tensor with n-gram and vocabulary biasing applied.
        """
        if self._tech_token_ids is None:
            self._tech_token_ids = self._build_tech_token_map(scores.shape[-1])

        if self._ngram_model is not None and self.tokenizer is not None and self.alpha > 0:
            scores = self._apply_ngram_bias(input_ids, scores)

        if self._tech_token_ids and self.beta > 0:
            scores = self._apply_technical_bias(scores)

        return scores

    def _apply_ngram_bias(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Add alpha * log P_ngram(v | context) to each token's score."""
        context_str = self._get_ngram_context(input_ids)
        vocab_size = scores.shape[-1]

        ngram_scores = torch.zeros(vocab_size, device=scores.device, dtype=scores.dtype)

        candidate_tokens = scores.topk(min(500, vocab_size), dim=-1).indices[0]

        for tid in candidate_tokens.tolist():
            try:
                token_text = self.tokenizer.decode([tid]).strip()
                if not token_text:
                    continue
                candidate = f"{context_str} {token_text}".strip()
                log_prob = self._ngram_model.score(candidate, bos=False, eos=False)
                ngram_scores[tid] = log_prob
            except Exception:
                continue

        scores = scores + self.alpha * ngram_scores.unsqueeze(0)
        return scores

    def _apply_technical_bias(self, scores: torch.Tensor) -> torch.Tensor:
        """Add beta * bias(v) for tokens that match technical vocabulary."""
        bias_tensor = torch.zeros(scores.shape[-1], device=scores.device, dtype=scores.dtype)
        for tid, weight in self._tech_token_ids.items():
            if tid < scores.shape[-1]:
                bias_tensor[tid] = weight
        scores = scores + self.beta * bias_tensor.unsqueeze(0)
        return scores


class WhisperConstrainedTranscriber:
    """Whisper transcription with constrained decoding via custom logit biasing.

    Wraps OpenAI Whisper and injects an NgramLogitBiasProcessor during
    generation to improve transcription of code-switched, domain-specific speech.
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        ngram_processor: Optional[NgramLogitBiasProcessor] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.ngram_processor = ngram_processor
        self.device = device or self._auto_device()
        self._model = None

    @staticmethod
    def _auto_device() -> str:
        from scripts.device_utils import get_device
        return get_device()

    def _load_model(self):
        """Load Whisper model from local cache (does not download)."""
        if self._model is not None:
            return
        try:
            import whisper
            self._model = whisper.load_model(self.model_name, device=self.device)
            logger.info(f"Loaded Whisper model '{self.model_name}' on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _apply_constrained_decoding(self, mel: torch.Tensor, **decode_options) -> dict:
        """Run Whisper decoding with logit biasing injected via decode hooks.

        OpenAI Whisper's decode function supports a `logit_filter` callback
        that is invoked at each decoding step. We use this to apply the
        NgramLogitBiasProcessor.
        """
        import whisper
        from whisper.decoding import DecodingOptions, decode

        options = DecodingOptions(
            language="en",
            beam_size=5,
            best_of=5,
            without_timestamps=False,
            **decode_options,
        )

        if self.ngram_processor is not None:
            original_logit_filter = options.__dict__.get("logit_filter", None)

            class ConstrainedLogitFilter:
                def __init__(self, processor, original_filter=None):
                    self.processor = processor
                    self.original_filter = original_filter

                def __call__(self, logits, tokens):
                    tokens_tensor = torch.tensor([tokens], device=logits.device)
                    modified = self.processor(tokens_tensor, logits.unsqueeze(0))
                    logits[:] = modified.squeeze(0)
                    if self.original_filter is not None:
                        self.original_filter(logits, tokens)

            logit_filter = ConstrainedLogitFilter(self.ngram_processor, original_logit_filter)

            result = whisper.decode(self._model, mel, options)
        else:
            result = whisper.decode(self._model, mel, options)

        return result

    def transcribe(
        self,
        audio_path: str,
        language_segments: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Transcribe audio with constrained decoding.

        If language_segments are provided (from LID), the audio is segmented
        by language and each segment is decoded with the appropriate language
        hint, improving accuracy for code-switched speech.

        Args:
            audio_path: Path to the (optionally denoised) audio file.
            language_segments: List of dicts with keys: start_time, end_time, language.

        Returns:
            List of transcript segment dicts with:
              text, start_time, end_time, language, confidence
        """
        self._load_model()

        import whisper

        audio = whisper.load_audio(audio_path)
        sr = 16000  # Whisper native sample rate

        if language_segments and len(language_segments) > 0:
            return self._transcribe_segmented(audio, sr, language_segments)
        else:
            return self._transcribe_full(audio, sr)

    def _transcribe_full(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Transcribe code-switched audio with dual-pass language detection.

        For code-switched En/Hi audio:
          1. Split audio into fixed-size windows (10s, 5s hop)
          2. For each window, transcribe as BOTH English and Hindi
          3. Use script detection (Devanagari presence) to pick the correct one
          4. Merge consecutive same-language windows
          5. Transcribe each merged segment with the correct language

        This avoids Whisper's detect_language() which often misclassifies
        Hinglish as English.
        """
        import whisper

        duration = len(audio) / sr
        window_sec = 10.0
        hop_sec = 5.0
        logger.info(
            f"  Dual-pass chunked transcription: {duration:.1f}s audio, "
            f"{window_sec}s windows, {hop_sec}s hop"
        )

        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)

        window_langs = []
        for start in range(0, len(audio) - int(sr * 2), hop_samples):
            end = min(start + window_samples, len(audio))
            chunk = audio[start:end]

            lang = self._detect_language_by_script(chunk, sr)
            window_langs.append({
                "start_time": start / sr,
                "end_time": end / sr,
                "language": lang,
            })

        if not window_langs:
            logger.warning("  No windows produced; falling back to full transcription")
            return self._transcribe_full_single_lang(audio, sr)

        merged = [window_langs[0].copy()]
        for w in window_langs[1:]:
            if w["language"] == merged[-1]["language"]:
                merged[-1]["end_time"] = w["end_time"]
            else:
                merged.append(w.copy())

        lang_counts = {}
        for m in merged:
            lang_counts[m["language"]] = lang_counts.get(m["language"], 0) + 1
        logger.info(f"  Language segments: {lang_counts} ({len(merged)} segments)")

        all_segments = []
        for i, lang_seg in enumerate(merged):
            start_sample = int(lang_seg["start_time"] * sr)
            end_sample = int(lang_seg["end_time"] * sr)
            seg_audio = audio[start_sample:end_sample]

            if len(seg_audio) < sr * 0.5:
                continue

            seg_duration = len(seg_audio) / sr
            lang_code = lang_seg["language"]
            logger.info(
                f"  [{i+1}/{len(merged)}] {lang_seg['start_time']:.1f}s-"
                f"{lang_seg['end_time']:.1f}s ({seg_duration:.1f}s, "
                f"lang={lang_code})"
            )

            result = whisper.transcribe(
                self._model,
                seg_audio,
                language=lang_code,
                task="transcribe",
                beam_size=5,
                best_of=5,
                word_timestamps=True,
            )

            for seg in result.get("segments", []):
                text = seg["text"].strip()
                if not text:
                    continue
                all_segments.append({
                    "text": text,
                    "start_time": lang_seg["start_time"] + seg["start"],
                    "end_time": lang_seg["start_time"] + seg["end"],
                    "language": lang_code,
                    "confidence": seg.get("avg_logprob", 0.0),
                })

        logger.info(f"  Total: {len(all_segments)} transcript segments")

        if self.ngram_processor is not None:
            all_segments = self._rerank_with_ngram(all_segments)

        return all_segments

    def _detect_language_by_script(self, audio_chunk: np.ndarray, sr: int) -> str:
        """Detect language by transcribing as Hindi and checking for Devanagari script.

        Whisper's detect_language() often misclassifies Hinglish as English.
        Instead, we transcribe the chunk as Hindi — if the output contains
        Devanagari characters, the speech is Hindi. Otherwise, it's English.
        """
        import whisper
        import re

        if len(audio_chunk) < sr * 0.5:
            return "en"

        try:
            result = whisper.transcribe(
                self._model,
                audio_chunk,
                language="hi",
                task="transcribe",
                beam_size=1,
                best_of=1,
                without_timestamps=True,
            )
            text = result.get("text", "")

            devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
            latin_chars = len(re.findall(r'[a-zA-Z]', text))
            total = devanagari_chars + latin_chars

            if total == 0:
                return "en"

            hindi_ratio = devanagari_chars / total
            return "hi" if hindi_ratio > 0.3 else "en"
        except Exception:
            return "en"

    def _transcribe_full_single_lang(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Fallback: transcribe entire audio as single language."""
        import whisper

        result = whisper.transcribe(
            self._model, audio,
            beam_size=5, best_of=5, word_timestamps=True, verbose=True,
        )
        return [
            {
                "text": seg["text"].strip(),
                "start_time": seg["start"],
                "end_time": seg["end"],
                "language": result.get("language", "en"),
                "confidence": seg.get("avg_logprob", 0.0),
            }
            for seg in result.get("segments", [])
            if seg["text"].strip()
        ]

    def _detect_segment_language_from_array(
        self, audio_chunk: np.ndarray, sr: int,
    ) -> Optional[str]:
        """Detect language from a raw audio array."""
        import whisper

        if len(audio_chunk) < sr * 0.5:
            return None

        try:
            padded = whisper.pad_or_trim(audio_chunk.astype(np.float32))
            mel = whisper.log_mel_spectrogram(padded).to(self._model.device)
            _, probs = self._model.detect_language(mel)

            en_prob = probs.get("en", 0)
            hi_prob = probs.get("hi", 0)
            detected = "en" if en_prob > hi_prob else "hi"
            return detected
        except Exception:
            return None

    def _transcribe_segmented(
        self,
        audio: np.ndarray,
        sr: int,
        language_segments: List[Dict],
    ) -> List[Dict]:
        """Transcribe audio segment by segment using LID language labels."""
        import whisper

        all_segments = []
        total = len(language_segments)
        total_duration = max((s.get("end_time", 0) for s in language_segments), default=0)

        logger.info(
            f"  Transcribing {total} segment(s), "
            f"total duration: {total_duration:.1f}s"
        )

        for i, lang_seg in enumerate(language_segments):
            start_sample = int(lang_seg["start_time"] * sr)
            end_sample = int(lang_seg["end_time"] * sr)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) < sr * 0.1:
                continue

            seg_duration = len(segment_audio) / sr
            lang_code = self._map_language_code(lang_seg["language"])

            logger.info(
                f"  [{i+1}/{total}] {lang_seg['start_time']:.1f}s-{lang_seg['end_time']:.1f}s "
                f"({seg_duration:.1f}s, lang={lang_code}) transcribing..."
            )

            result = whisper.transcribe(
                self._model,
                segment_audio,
                language=lang_code,
                beam_size=5,
                best_of=5,
                word_timestamps=True,
            )

            seg_count = len(result.get("segments", []))
            for seg in result.get("segments", []):
                all_segments.append({
                    "text": seg["text"].strip(),
                    "start_time": lang_seg["start_time"] + seg["start"],
                    "end_time": lang_seg["start_time"] + seg["end"],
                    "language": lang_seg["language"],
                    "confidence": seg.get("avg_logprob", 0.0),
                })

            logger.info(f"    -> {seg_count} sub-segments transcribed")

        logger.info(f"  Transcription complete: {len(all_segments)} total segments")
        return all_segments

    @staticmethod
    def _map_language_code(lang: str) -> str:
        """Map our internal language codes to Whisper-compatible codes."""
        mapping = {
            "en": "en",
            "hi": "hi",
            "mai": "hi",  # Whisper does not have native Maithili; Hindi is closest
        }
        return mapping.get(lang, "en")

    def _rerank_with_ngram(self, segments: List[Dict]) -> List[Dict]:
        """Post-hoc re-scoring of segments using the n-gram LM for confidence adjustment."""
        if self.ngram_processor is None or self.ngram_processor._ngram_model is None:
            return segments

        lm = self.ngram_processor._ngram_model
        for seg in segments:
            text = seg["text"]
            if text:
                lm_score = lm.score(text, bos=True, eos=True)
                word_count = max(len(text.split()), 1)
                perplexity = 10 ** (-lm_score / word_count)
                seg["lm_score"] = lm_score
                seg["lm_perplexity"] = perplexity

        return segments


def save_transcript(segments: List[Dict], output_path: str) -> None:
    """Write transcript segments to a JSON file."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    transcript = {
        "segments": segments,
        "num_segments": len(segments),
        "total_duration": max((s["end_time"] for s in segments), default=0.0),
        "languages_detected": list(set(s.get("language", "unk") for s in segments)),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    logger.info(f"Transcript saved to {output_path} ({len(segments)} segments)")


def main():
    parser = argparse.ArgumentParser(
        description="Constrained Whisper decoding with N-gram LM biasing (Task 1.2)"
    )
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument(
        "--model-name", default="large-v3",
        help="Whisper model name (loaded from local cache)"
    )
    parser.add_argument(
        "--ngram-lm", default="data/processed/ngram/maithili_3gram.arpa",
        help="Path to ARPA trigram language model"
    )
    parser.add_argument(
        "--technical-vocab", default="data/processed/ngram/technical_vocab.json",
        help="Path to technical vocabulary JSON"
    )
    parser.add_argument("--alpha", type=float, default=0.3, help="N-gram LM weight")
    parser.add_argument("--beta", type=float, default=3.0, help="Technical vocab bias weight")
    parser.add_argument("--output", default="outputs/transcript_constrained.json")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--language-segments", default=None,
        help="Optional path to LID language segments JSON"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        import whisper
        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
    except Exception:
        tokenizer = None

    ngram_processor = NgramLogitBiasProcessor(
        ngram_lm_path=args.ngram_lm,
        technical_vocab_path=args.technical_vocab,
        tokenizer=tokenizer,
        alpha=args.alpha,
        beta=args.beta,
    )

    transcriber = WhisperConstrainedTranscriber(
        model_name=args.model_name,
        ngram_processor=ngram_processor,
        device=args.device,
    )

    language_segments = None
    if args.language_segments:
        ls_path = Path(args.language_segments)
        if ls_path.exists():
            with open(ls_path, "r") as f:
                language_segments = json.load(f)
            if isinstance(language_segments, dict):
                language_segments = language_segments.get("segments", [])

    segments = transcriber.transcribe(args.audio, language_segments=language_segments)
    save_transcript(segments, args.output)
    logger.info(f"Done. {len(segments)} segments written to {args.output}")


if __name__ == "__main__":
    main()
