"""
Code-switch-aware Hinglish-to-IPA converter for Task 2.1.

Converts a code-switched transcript (En/Hi/Mai) into IPA using:
  - epitran (hin-Deva) for Hindi/Maithili Devanagari text
  - eng_to_ipa for English text
  - Maithili-specific phoneme overrides from a G2P JSON file
  - Break marker '|' at code-switch boundaries

Expected input: outputs/transcript_code_switched.json (from Part I pipeline)
Expected output: outputs/transcript_ipa.json
"""

import json
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MAITHILI_OVERRIDES_DEFAULT: Dict[str, str] = {
    "ऐ": "əɪ",
    "औ": "əʊ",
    "ऋ": "ɾɪ",
    "ण": "ɳ",
    "ष": "ʂ",
    "क्ष": "kʂ",
    "त्र": "t̪ɾ",
    "ज्ञ": "ɡjə",
    "श": "ʃ",
    "छ": "t͡ʃʰ",
    "झ": "d͡ʒʰ",
    "ढ": "ɖʱ",
    "ठ": "ʈʰ",
    "ङ": "ŋ",
    "ञ": "ɲ",
    "अं": "əŋ",
    "अः": "əh",
}

DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")


class HinglishIPAConverter:
    """Converts code-switched Hinglish (+ Maithili) text to IPA.

    Uses epitran for Hindi Devanagari, eng_to_ipa for English,
    and applies Maithili-specific overrides for vowel diphthongs
    and retroflex consonants where Maithili diverges from Hindi.
    """

    def __init__(self, g2p_overrides_path: Optional[str] = None):
        """
        Args:
            g2p_overrides_path: Path to a JSON mapping Devanagari graphemes
                to Maithili IPA (keys = grapheme, values = IPA without slashes).
                Falls back to hardcoded MAITHILI_OVERRIDES_DEFAULT.
        """
        import epitran
        import eng_to_ipa

        self._epi = epitran.Epitran("hin-Deva")
        self._eng_to_ipa = eng_to_ipa

        self._overrides = dict(MAITHILI_OVERRIDES_DEFAULT)
        if g2p_overrides_path:
            self._load_overrides(g2p_overrides_path)

        self._override_pairs = self._build_sorted_override_pairs()

    def _load_overrides(self, path: str) -> None:
        """Load Maithili G2P overrides from a JSON file.

        The JSON can be either:
          - A flat dict: {"ऐ": "əɪ", ...}  (grapheme → IPA)
          - A dict of word → IPA mappings produced by prepare_translation_data.py
        Only entries whose keys are short Devanagari graphemes (≤4 chars) are
        treated as phoneme overrides; longer entries are ignored here since
        they represent full word mappings handled at the word level.
        """
        p = Path(path)
        if not p.exists():
            logger.warning(f"G2P overrides file not found: {path}")
            return
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key, val in data.items():
            clean_val = val.strip("/").strip()
            if len(key) <= 4 and DEVANAGARI_RANGE.search(key):
                self._overrides[key] = clean_val

        logger.info(
            f"Loaded G2P overrides from {path} "
            f"({len(self._overrides)} active phoneme overrides)"
        )

    def _build_sorted_override_pairs(self) -> List[tuple]:
        """Sort overrides longest-first so multi-char conjuncts match before singles."""
        return sorted(self._overrides.items(), key=lambda kv: -len(kv[0]))

    def convert_english(self, text: str) -> str:
        """Convert English text to IPA via eng_to_ipa."""
        result = self._eng_to_ipa.convert(text)
        result = result.replace("*", "")
        return result.strip()

    def convert_hindi(self, text: str) -> str:
        """Convert Hindi/Maithili Devanagari text to IPA via epitran + overrides.

        Multi-character conjuncts (क्ष, त्र, ज्ञ) are replaced before
        single-character overrides to avoid partial matches.
        """
        base_ipa = self._epi.transliterate(text)
        for grapheme, ipa_replacement in self._override_pairs:
            grapheme_ipa = self._epi.transliterate(grapheme)
            if grapheme_ipa and grapheme_ipa in base_ipa:
                base_ipa = base_ipa.replace(grapheme_ipa, ipa_replacement)
        return base_ipa.strip()

    def convert_segment(self, text: str, language: str) -> str:
        """Route text to the appropriate G2P converter based on language tag.

        Args:
            text: The segment text.
            language: One of "en", "hi", "mai", or "unk".

        Returns:
            IPA string for the segment.
        """
        if not text or not text.strip():
            return ""

        lang = language.lower().strip()

        if lang in ("en", "eng", "english"):
            return self.convert_english(text)
        elif lang in ("hi", "hin", "hindi", "mai", "maithili"):
            return self.convert_hindi(text)
        else:
            if DEVANAGARI_RANGE.search(text):
                return self.convert_hindi(text)
            return self.convert_english(text)

    def convert_transcript(
        self,
        transcript_path: str,
        output_path: str,
    ) -> Dict:
        """Process a full code-switched transcript JSON and add IPA fields.

        Reads the transcript produced by Part I (run_stt_pipeline.py), which
        has the structure:
          { "segments": [ { "start_time", "end_time", "text", "language", ... }, ... ] }

        For each segment, adds an "ipa" field. At code-switch boundaries
        between consecutive segments, inserts a '|' break marker.

        Args:
            transcript_path: Path to transcript_code_switched.json.
            output_path: Path to write transcript_ipa.json.

        Returns:
            Summary dict with counts and output path.
        """
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        segments = transcript.get("segments", [])
        if not segments and isinstance(transcript, list):
            segments = transcript

        prev_lang = None
        ipa_segments = []

        for seg in segments:
            text = seg.get("text", "")
            language = seg.get("language", "unk")

            ipa = self.convert_segment(text, language)

            is_switch = prev_lang is not None and language != prev_lang
            if is_switch and ipa_segments:
                ipa_segments[-1]["ipa"] += " |"

            ipa_seg = {
                "start_time": seg.get("start_time", 0.0),
                "end_time": seg.get("end_time", 0.0),
                "text": text,
                "language": language,
                "ipa": ipa,
                "code_switch_before": is_switch,
            }
            ipa_segments.append(ipa_seg)
            prev_lang = language

        output_data = {
            "source_transcript": transcript_path,
            "num_segments": len(ipa_segments),
            "segments": ipa_segments,
        }

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"IPA transcript written to {output_path} "
            f"({len(ipa_segments)} segments, "
            f"{sum(1 for s in ipa_segments if s['code_switch_before'])} code-switches)"
        )

        return {
            "output_path": str(out_p),
            "num_segments": len(ipa_segments),
            "code_switches": sum(1 for s in ipa_segments if s["code_switch_before"]),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Code-switch-aware Hinglish-to-IPA converter (Task 2.1)"
    )
    parser.add_argument(
        "--input",
        default="outputs/transcript_code_switched.json",
        help="Path to code-switched transcript JSON from Part I",
    )
    parser.add_argument(
        "--output",
        default="outputs/transcript_ipa.json",
        help="Path to write IPA-annotated transcript JSON",
    )
    parser.add_argument(
        "--g2p-overrides",
        default="data/processed/translation/maithili_grapheme_to_ipa.json",
        help="Path to Maithili G2P overrides JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    converter = HinglishIPAConverter(g2p_overrides_path=args.g2p_overrides)
    result = converter.convert_transcript(args.input, args.output)
    logger.info(f"IPA conversion complete: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
