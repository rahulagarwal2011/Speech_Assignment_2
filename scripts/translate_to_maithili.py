"""
Maithili translator using IndicTrans2 + technical dictionary for Task 2.2.

Translates a code-switched transcript (En/Hi) to Maithili using:
  - IndicTrans2 en-indic-1B for English segments
  - IndicTrans2 indic-indic-1B for Hindi segments
  - Technical dictionary lookup for domain-specific terms (priority over model)

Expected input:  outputs/transcript_code_switched.json (from Part I pipeline)
Expected output: outputs/maithili_translated_text.json
"""

import csv
import json
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MaithiliTranslator:
    """Translate code-switched En/Hi text to Maithili via IndicTrans2.

    Translation strategy per segment:
      1. Scan for technical terms in the dictionary and cache their Maithili equivalents.
      2. Replace technical terms with placeholder tokens before model inference.
      3. Run IndicTrans2 on the remaining text.
      4. Restore technical term translations in the output.
    """

    PLACEHOLDER_TPL = "TECHTERM{idx}"
    PLACEHOLDER_RE = re.compile(r"TECHTERM\{(\d+)\}")

    def __init__(
        self,
        en_model_name: str = "ai4bharat/indictrans2-en-indic-1B",
        hi_model_name: str = "ai4bharat/indictrans2-indic-indic-1B",
        dictionary_path: Optional[str] = None,
    ):
        """
        Args:
            en_model_name: HuggingFace model ID for English→Indic IndicTrans2.
            hi_model_name: HuggingFace model ID for Indic→Indic IndicTrans2.
            dictionary_path: Path to TSV technical dictionary with columns
                english_term, hindi_equivalent, maithili, ipa_maithili, domain.
        """
        self.en_model_name = en_model_name
        self.hi_model_name = hi_model_name

        self._en_model = None
        self._en_tokenizer = None
        self._hi_model = None
        self._hi_tokenizer = None

        self._dictionary: Dict[str, Dict[str, str]] = {}
        if dictionary_path:
            self._load_dictionary(dictionary_path)

    def _load_dictionary(self, path: str) -> None:
        """Load technical term TSV dictionary.

        Expected columns: english_term, hindi_equivalent, maithili, ipa_maithili, domain
        Only entries with a non-empty 'maithili' value are loaded.
        """
        p = Path(path)
        if not p.exists():
            logger.warning(f"Dictionary not found: {path}")
            return

        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                en = (row.get("english_term") or row.get("english", "")).strip().lower()
                mai = (row.get("maithili") or "").strip()
                if en and mai and not mai.startswith("[NEEDS_"):
                    self._dictionary[en] = {
                        "maithili": mai,
                        "hindi": (row.get("hindi_equivalent") or "").strip(),
                        "ipa": (row.get("ipa_maithili") or "").strip(),
                        "domain": (row.get("domain") or "").strip(),
                    }

        logger.info(f"Loaded {len(self._dictionary)} terms from dictionary at {path}")

    def _load_en_model(self) -> None:
        """Lazy-load the English->Indic IndicTrans2 model with IndicTransToolkit."""
        if self._en_model is not None:
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit import IndicProcessor

        logger.info(f"Loading English->Indic model: {self.en_model_name}")
        self._en_tokenizer = AutoTokenizer.from_pretrained(
            self.en_model_name, trust_remote_code=True,
        )
        self._en_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.en_model_name, trust_remote_code=True,
        )
        self._en_model.eval()
        if not hasattr(self, "_ip"):
            self._ip = IndicProcessor(inference=True)

    def _load_hi_model(self) -> None:
        """Lazy-load the Indic->Indic IndicTrans2 model with IndicTransToolkit."""
        if self._hi_model is not None:
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit import IndicProcessor

        logger.info(f"Loading Indic->Indic model: {self.hi_model_name}")
        self._hi_tokenizer = AutoTokenizer.from_pretrained(
            self.hi_model_name, trust_remote_code=True,
        )
        self._hi_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hi_model_name, trust_remote_code=True,
        )
        self._hi_model.eval()
        if not hasattr(self, "_ip"):
            self._ip = IndicProcessor(inference=True)

    def lookup_technical_terms(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Scan text for known technical terms and return replacements.

        Returns:
            (modified_text, matched_terms) where modified_text has technical
            terms replaced with TECHTERM{N} placeholders, and matched_terms
            is a list of dicts with 'english', 'maithili', 'placeholder' keys.
        """
        if not self._dictionary:
            return text, []

        matched = []
        text_lower = text.lower()
        sorted_terms = sorted(self._dictionary.keys(), key=len, reverse=True)

        for term in sorted_terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(text_lower):
                idx = len(matched)
                placeholder = self.PLACEHOLDER_TPL.format(idx=idx)
                matched.append({
                    "english": term,
                    "maithili": self._dictionary[term]["maithili"],
                    "placeholder": placeholder,
                })
                text = pattern.sub(placeholder, text, count=1)
                text_lower = text.lower()

        return text, matched

    def _restore_technical_terms(
        self, translated: str, matched_terms: List[Dict[str, str]]
    ) -> str:
        """Replace placeholders back with Maithili technical term translations."""
        for entry in matched_terms:
            translated = translated.replace(
                entry["placeholder"], entry["maithili"]
            )
        translated = self.PLACEHOLDER_RE.sub("", translated)
        return translated.strip()

    def translate_english(self, text: str) -> str:
        """Translate English text to Maithili via IndicTrans2 en->indic."""
        self._load_en_model()

        batch = self._ip.preprocess_batch(
            [text], src_lang="eng_Latn", tgt_lang="mai_Deva",
        )
        inputs = self._en_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=256,
        )

        import torch
        with torch.no_grad():
            generated = self._en_model.generate(
                **inputs, max_length=256, num_beams=5,
            )

        decoded = self._en_tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True,
        )
        outputs = self._ip.postprocess_batch(decoded, lang="mai_Deva")
        return outputs[0].strip() if outputs else ""

    def translate_hindi(self, text: str) -> str:
        """Translate Hindi text to Maithili via IndicTrans2 indic->indic."""
        self._load_hi_model()

        batch = self._ip.preprocess_batch(
            [text], src_lang="hin_Deva", tgt_lang="mai_Deva",
        )
        inputs = self._hi_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=256,
        )

        import torch
        with torch.no_grad():
            generated = self._hi_model.generate(
                **inputs, max_length=256, num_beams=5,
            )

        decoded = self._hi_tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True,
        )
        outputs = self._ip.postprocess_batch(decoded, lang="mai_Deva")
        return outputs[0].strip() if outputs else ""

    def translate_segment(self, text: str, source_lang: str) -> Dict:
        """Translate a single segment: dictionary first, then model fallback.

        Args:
            text: Source text.
            source_lang: Language tag — "en", "hi", "mai", etc.

        Returns:
            Dict with keys: maithili_text, translation_method, technical_terms_used.
        """
        if not text or not text.strip():
            return {
                "maithili_text": "",
                "translation_method": "empty",
                "technical_terms_used": [],
            }

        lang = source_lang.lower().strip()

        if lang in ("mai", "maithili"):
            return {
                "maithili_text": text,
                "translation_method": "passthrough",
                "technical_terms_used": [],
            }

        modified_text, matched_terms = self.lookup_technical_terms(text)

        method_parts = []
        if matched_terms:
            method_parts.append("dictionary")

        has_remaining_text = bool(
            re.sub(r"TECHTERM\{\d+\}", "", modified_text).strip()
        )

        if has_remaining_text:
            if lang in ("en", "eng", "english"):
                translated = self.translate_english(modified_text)
                method_parts.append("indictrans2_en")
            elif lang in ("hi", "hin", "hindi"):
                translated = self.translate_hindi(modified_text)
                method_parts.append("indictrans2_hi")
            else:
                translated = self.translate_english(modified_text)
                method_parts.append("indictrans2_en_fallback")
        else:
            translated = modified_text

        final_text = self._restore_technical_terms(translated, matched_terms)

        return {
            "maithili_text": final_text,
            "translation_method": "+".join(method_parts) if method_parts else "none",
            "technical_terms_used": [
                {"english": m["english"], "maithili": m["maithili"]}
                for m in matched_terms
            ],
        }

    def translate_transcript(
        self, transcript_path: str, output_path: str,
    ) -> Dict:
        """Translate an entire code-switched transcript to Maithili.

        Reads the transcript JSON (Part I output), translates each segment,
        preserves timing information, and writes the result.

        Args:
            transcript_path: Path to transcript_code_switched.json.
            output_path: Path to write maithili_translated_text.json.

        Returns:
            Summary dict.
        """
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        segments = transcript.get("segments", [])
        if not segments and isinstance(transcript, list):
            segments = transcript

        translated_segments = []
        method_counts: Dict[str, int] = {}

        for seg in segments:
            text = seg.get("text", "")
            lang = seg.get("language", "en")

            result = self.translate_segment(text, lang)

            translated_seg = {
                "start_time": seg.get("start_time", 0.0),
                "end_time": seg.get("end_time", 0.0),
                "original_text": text,
                "original_language": lang,
                "maithili_text": result["maithili_text"],
                "translation_method": result["translation_method"],
                "technical_terms_used": result["technical_terms_used"],
            }
            translated_segments.append(translated_seg)

            method = result["translation_method"]
            method_counts[method] = method_counts.get(method, 0) + 1

        output_data = {
            "source_transcript": transcript_path,
            "num_segments": len(translated_segments),
            "method_summary": method_counts,
            "segments": translated_segments,
        }

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Maithili translation written to {output_path} "
            f"({len(translated_segments)} segments)"
        )
        logger.info(f"Translation method counts: {method_counts}")

        return {
            "output_path": str(out_p),
            "num_segments": len(translated_segments),
            "method_summary": method_counts,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Translate code-switched transcript to Maithili (Task 2.2)"
    )
    parser.add_argument(
        "--input",
        default="outputs/transcript_code_switched.json",
        help="Path to code-switched transcript JSON from Part I",
    )
    parser.add_argument(
        "--output",
        default="outputs/maithili_translated_text.json",
        help="Path to write Maithili translated transcript JSON",
    )
    parser.add_argument(
        "--dictionary",
        default="data/manual/technical_parallel_corpus.tsv",
        help="Path to technical dictionary TSV",
    )
    parser.add_argument(
        "--en-model",
        default="ai4bharat/indictrans2-en-indic-1B",
        help="HuggingFace model ID for English→Indic IndicTrans2",
    )
    parser.add_argument(
        "--hi-model",
        default="ai4bharat/indictrans2-indic-indic-1B",
        help="HuggingFace model ID for Indic→Indic IndicTrans2",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    translator = MaithiliTranslator(
        en_model_name=args.en_model,
        hi_model_name=args.hi_model,
        dictionary_path=args.dictionary,
    )
    result = translator.translate_transcript(args.input, args.output)
    logger.info(f"Translation complete: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
