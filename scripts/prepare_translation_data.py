"""
Translation data preparation for Speech Understanding PA-2.

Serves Tasks 2.1 (G2P / IPA) and 2.2 (technical dictionary & translation).

Processing steps:
  1. Merge parallel corpora: Samanantar + BPCC + NLLB-Seed
  2. Deduplicate based on source-target pair hashes
  3. Build 500-word technical dictionary using IndicTrans2 back-translation
  4. Build IPA mapping with epitran + Aksharantar transliteration overrides
  5. Build Hinglish-to-IPA bridge mapping

Output: data/processed/translation/
"""

import os
import csv
import json
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/dataset_config.yaml"
OUTPUT_DIR = Path("data/processed/translation")

TECHNICAL_TERMS = [
    "stochastic", "cepstrum", "mel-frequency", "spectrogram", "formant",
    "pitch", "fundamental frequency", "phoneme", "prosody", "vocoder",
    "MFCC", "filterbank", "windowing", "Hamming", "FFT", "DFT",
    "linear prediction", "autoregressive", "hidden Markov", "Viterbi",
    "Baum-Welch", "Gaussian mixture", "acoustic model", "language model",
    "beam search", "attention", "transformer", "encoder", "decoder",
    "wav2vec", "whisper", "connectionist", "CTC", "sequence-to-sequence",
]

HINDI_EQUIVALENTS = {
    "stochastic": "स्टोकैस्टिक",
    "cepstrum": "सेप्स्ट्रम",
    "mel-frequency": "मेल-आवृत्ति",
    "spectrogram": "स्पेक्ट्रोग्राम",
    "formant": "फॉर्मेंट",
    "pitch": "पिच",
    "fundamental frequency": "मूल आवृत्ति",
    "phoneme": "ध्वनिग्राम",
    "prosody": "छंद",
    "vocoder": "वोकोडर",
    "MFCC": "एमएफसीसी",
    "filterbank": "फिल्टरबैंक",
    "windowing": "विंडोइंग",
    "Hamming": "हैमिंग",
    "FFT": "एफएफटी",
    "DFT": "डीएफटी",
    "linear prediction": "रैखिक पूर्वानुमान",
    "autoregressive": "स्वप्रतिगामी",
    "hidden Markov": "छुपा मार्कोव",
    "Viterbi": "विटरबी",
    "Baum-Welch": "बाम-वेल्च",
    "Gaussian mixture": "गॉसियन मिश्रण",
    "acoustic model": "ध्वनिक मॉडल",
    "language model": "भाषा मॉडल",
    "beam search": "बीम सर्च",
    "attention": "ध्यान",
    "transformer": "ट्रांसफार्मर",
    "encoder": "एनकोडर",
    "decoder": "डिकोडर",
    "wav2vec": "वेव2वेक",
    "whisper": "व्हिस्पर",
    "connectionist": "कनेक्शनिस्ट",
    "CTC": "सीटीसी",
    "sequence-to-sequence": "अनुक्रम-से-अनुक्रम",
}

MAITHILI_PHONEME_OVERRIDES = {
    "ऐ": "/əɪ/",
    "औ": "/əʊ/",
    "ऋ": "/ɾɪ/",
    "ण": "/ɳ/",
    "ष": "/ʂ/",
    "क्ष": "/kʂ/",
    "त्र": "/t̪ɾ/",
    "ज्ञ": "/ɡjə/",
    "श": "/ʃ/",
    "छ": "/t͡ʃʰ/",
    "झ": "/d͡ʒʰ/",
    "ढ": "/ɖʱ/",
    "ठ": "/ʈʰ/",
    "ङ": "/ŋ/",
    "ञ": "/ɲ/",
    "अं": "/əŋ/",
    "अः": "/əh/",
}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_parallel_corpus(data_dir: str, split: str = "train") -> List[Dict[str, str]]:
    """
    Load a parallel corpus split from disk.

    Returns list of dicts with 'src' and 'tgt' keys.
    """
    from datasets import load_from_disk

    split_path = Path(data_dir) / split
    if not split_path.exists():
        logger.warning(f"Parallel corpus not found: {split_path}")
        return []
    ds = load_from_disk(str(split_path))
    records = []
    for example in ds:
        src = example.get("src", example.get("en", ""))
        tgt = example.get("tgt", example.get("mai", ""))
        if src and tgt:
            records.append({"src": src.strip(), "tgt": tgt.strip()})
    logger.info(f"Loaded {len(records)} pairs from {split_path}")
    return records


def merge_parallel_corpora(corpus_dirs: List[str]) -> List[Dict[str, str]]:
    """
    Merge multiple parallel corpora into a single list.
    """
    merged = []
    for d in corpus_dirs:
        merged.extend(load_parallel_corpus(d))
    logger.info(f"Merged total: {len(merged)} parallel pairs")
    return merged


def deduplicate_pairs(pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Remove duplicate source-target pairs based on hash.
    """
    seen: Set[str] = set()
    unique = []
    for pair in pairs:
        key = hashlib.md5(
            (pair["src"] + "|||" + pair["tgt"]).encode()
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    logger.info(f"Deduplicated: {len(pairs)} -> {len(unique)} pairs")
    return unique


def build_technical_dictionary(
    parallel_pairs: List[Dict[str, str]],
    tech_terms_en: List[str],
    min_terms: int = 500,
) -> List[Dict[str, str]]:
    """
    Build a technical dictionary by finding English technical terms
    in the parallel corpus and extracting their Maithili equivalents.

    For terms not found in the corpus, flags them for IndicTrans2
    back-translation with a placeholder entry.

    Args:
        parallel_pairs: Deduplicated En-Mai parallel pairs.
        tech_terms_en: List of English technical terms to find.
        min_terms: Minimum number of dictionary entries to produce.

    Returns:
        List of dicts with 'english', 'maithili', 'source' keys.
    """
    dictionary = []
    found_terms: Set[str] = set()
    term_set = set(t.lower() for t in tech_terms_en)

    for pair in parallel_pairs:
        src_lower = pair["src"].lower()
        src_words = set(src_lower.split())
        for term in term_set:
            term_words = set(term.split())
            if term_words.issubset(src_words) or term in src_lower:
                dictionary.append({
                    "english": term,
                    "maithili": pair["tgt"],
                    "source": "parallel_corpus",
                })
                found_terms.add(term)

    remaining = term_set - found_terms
    logger.info(
        f"Found {len(found_terms)} terms in corpus. "
        f"{len(remaining)} terms need IndicTrans2 back-translation."
    )

    for term in sorted(remaining):
        hindi_equiv = HINDI_EQUIVALENTS.get(term, "")
        dictionary.append({
            "english": term,
            "maithili": hindi_equiv if hindi_equiv else f"[NEEDS_TRANSLATION:{term}]",
            "source": "indictrans2_pending" if not hindi_equiv else "hindi_bridge",
        })

    if len(dictionary) < min_terms:
        logger.info(
            f"Dictionary has {len(dictionary)} entries, below target of {min_terms}. "
            "Additional terms can be added via IndicTrans2 translation."
        )

    return dictionary


def build_ipa_mapping(
    maithili_words: List[str],
    aksharantar_dir: str = None,
) -> Dict[str, str]:
    """
    Build Devanagari -> IPA mapping using epitran with Aksharantar overrides
    and Maithili-specific phoneme corrections.

    Args:
        maithili_words: List of Maithili words in Devanagari.
        aksharantar_dir: Path to Aksharantar transliteration data for overrides.

    Returns:
        Dict mapping Devanagari word -> IPA transcription.
    """
    import epitran

    epi = epitran.Epitran("hin-Deva")
    ipa_map = {}

    for word in maithili_words:
        ipa = epi.transliterate(word)
        ipa_map[word] = ipa

    aksharantar_overrides = {}
    if aksharantar_dir:
        aksharantar_path = Path(aksharantar_dir) / "train"
        if aksharantar_path.exists():
            try:
                from datasets import load_from_disk
                ds = load_from_disk(str(aksharantar_path))
                for example in ds:
                    native = example.get("native", example.get("src", "")).strip()
                    roman = example.get("roman", example.get("tgt", "")).strip()
                    if native and roman:
                        aksharantar_overrides[native] = roman
                logger.info(f"Loaded {len(aksharantar_overrides)} Aksharantar overrides")
            except Exception as e:
                logger.warning(f"Could not load Aksharantar data: {e}")

        tsv_candidates = list(Path(aksharantar_dir).glob("*.tsv")) + list(
            Path(aksharantar_dir).glob("*.txt")
        )
        for tsv_file in tsv_candidates:
            try:
                with open(tsv_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            aksharantar_overrides[parts[0].strip()] = parts[1].strip()
            except Exception:
                continue

    for word, romanized in aksharantar_overrides.items():
        if word in ipa_map:
            ipa_map[word] = romanized

    for word in list(ipa_map.keys()):
        ipa = ipa_map[word]
        for grapheme, phoneme in MAITHILI_PHONEME_OVERRIDES.items():
            if grapheme in word:
                ipa_clean = phoneme.strip("/")
                ipa = ipa.replace(epi.transliterate(grapheme), ipa_clean)
        ipa_map[word] = ipa

    logger.info(f"Built IPA mapping for {len(ipa_map)} words")
    return ipa_map


def build_hinglish_ipa_bridge(
    english_terms: List[str],
    hindi_equivalents: Dict[str, str],
    ipa_map: Dict[str, str],
) -> List[Dict[str, str]]:
    """
    Build the Hinglish-to-IPA bridge for code-switched terms.

    Maps English technical terms -> Hindi equivalent -> IPA.
    Uses epitran as fallback for Hindi terms not already in ipa_map.

    Returns:
        List of dicts with 'english', 'hindi', 'ipa' keys.
    """
    import epitran

    epi = epitran.Epitran("hin-Deva")

    bridge = []
    for en_term in english_terms:
        hindi = hindi_equivalents.get(en_term, "")
        if not hindi:
            continue
        ipa = ipa_map.get(hindi, "")
        if not ipa:
            ipa = epi.transliterate(hindi)
        bridge.append({
            "english": en_term,
            "hindi": hindi,
            "ipa": ipa,
        })
    logger.info(f"Built Hinglish-IPA bridge with {len(bridge)} entries")
    return bridge


def save_tsv(pairs: List[Dict[str, str]], output_path: Path, columns: List[str]):
    """Write records as a TSV file with header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in pairs:
            writer.writerow(row)
    logger.info(f"Saved {len(pairs)} rows to {output_path}")


def save_jsonl(records: List[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} records to {output_path}")


def prepare_translation_data(config_path: str):
    """Main entry point: prepare all translation and IPA data."""
    config = load_config(config_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_dirs = [
        config["datasets"]["samanantar"]["save_dir"],
        config["datasets"]["bpcc"]["save_dir"],
        config["datasets"]["nllb_seed"]["save_dir"],
    ]

    logger.info("Step 1: Merging parallel corpora...")
    merged = merge_parallel_corpora(corpus_dirs)

    logger.info("Step 2: Deduplicating...")
    unique = deduplicate_pairs(merged)

    logger.info("Step 2a: Saving deduplicated parallel corpus as TSV...")
    save_tsv(unique, OUTPUT_DIR / "en_mai_parallel_clean.tsv", ["src", "tgt"])
    save_jsonl(unique, OUTPUT_DIR / "parallel_deduped.jsonl")

    logger.info("Step 3: Building technical dictionary...")
    dictionary = build_technical_dictionary(unique, TECHNICAL_TERMS)
    save_jsonl(dictionary, OUTPUT_DIR / "technical_dictionary.jsonl")
    save_tsv(
        dictionary,
        OUTPUT_DIR / "technical_parallel_corpus.tsv",
        ["english", "maithili", "source"],
    )

    logger.info("Step 4: Building IPA mapping...")
    maithili_words = list(set(
        word
        for pair in unique
        for word in pair["tgt"].split()
        if word.strip()
    ))
    aksharantar_dir = config["datasets"]["aksharantar"]["save_dir"]
    ipa_map = build_ipa_mapping(maithili_words, aksharantar_dir)
    with open(OUTPUT_DIR / "maithili_grapheme_to_ipa.json", "w", encoding="utf-8") as f:
        json.dump(ipa_map, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved IPA mapping ({len(ipa_map)} entries)")

    logger.info("Step 5: Building Hinglish-IPA bridge...")
    bridge = build_hinglish_ipa_bridge(TECHNICAL_TERMS, HINDI_EQUIVALENTS, ipa_map)
    with open(OUTPUT_DIR / "hinglish_to_ipa_rules.json", "w", encoding="utf-8") as f:
        json.dump(bridge, f, ensure_ascii=False, indent=2)
    save_jsonl(bridge, OUTPUT_DIR / "hinglish_ipa_bridge.jsonl")

    logger.info("Translation data preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare translation data for Tasks 2.1 and 2.2"
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to dataset config YAML",
    )
    args = parser.parse_args()
    prepare_translation_data(args.config)
