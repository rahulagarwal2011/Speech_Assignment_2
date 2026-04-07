"""
N-gram language model data preparation for Speech Understanding PA-2.

Serves Task 1.2: building a Maithili-aware N-gram language model with
technical vocabulary logit bias for constrained decoding.

Processing steps:
  1. Load Sangraha monolingual Maithili text
  2. Tokenize with indicnlp sentence/word tokenizer
  3. Build unigram, bigram, and trigram frequency tables
  4. Create technical vocabulary list from the manual dictionary
  5. Assign logit bias weights to technical terms
  6. Build ARPA-format LM via KenLM
  7. Export logit bias dictionary as JSON

Output: data/processed/ngram/
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from collections import Counter

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/dataset_config.yaml"
OUTPUT_DIR = Path("data/processed/ngram")

TECHNICAL_TERMS = [
    "stochastic", "cepstrum", "mel-frequency", "spectrogram", "formant",
    "pitch", "fundamental frequency", "phoneme", "prosody", "vocoder",
    "MFCC", "filterbank", "windowing", "Hamming", "FFT", "DFT",
    "linear prediction", "autoregressive", "hidden Markov", "Viterbi",
    "Baum-Welch", "Gaussian mixture", "acoustic model", "language model",
    "beam search", "attention", "transformer", "encoder", "decoder",
    "wav2vec", "whisper", "connectionist", "CTC", "sequence-to-sequence",
]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sangraha_text(data_dir: str) -> List[str]:
    """
    Load Sangraha monolingual Maithili text corpus.

    Returns list of text strings (one per document/sentence).
    """
    from datasets import load_from_disk

    texts = []
    split_path = Path(data_dir) / "train"
    if not split_path.exists():
        logger.warning(f"Sangraha data not found: {split_path}")
        return texts
    ds = load_from_disk(str(split_path))
    for example in ds:
        text = example.get("text", "")
        if text.strip():
            texts.append(text.strip())
    logger.info(f"Loaded {len(texts)} Sangraha text documents")
    return texts


def tokenize_indicnlp(texts: List[str], lang: str = "mai") -> List[List[str]]:
    """
    Tokenize texts using the Indic NLP Library tokenizer.

    Args:
        texts: List of raw text strings.
        lang: ISO 639-3 language code.

    Returns:
        List of tokenized sentences (each a list of tokens).
    """
    from indicnlp.tokenize import indic_tokenize

    tokenized = []
    for text in texts:
        tokens = list(indic_tokenize.trivial_tokenize(text, lang))
        if tokens:
            tokenized.append(tokens)
    logger.info(f"Tokenized {len(tokenized)} sentences")
    return tokenized


def build_ngram_tables(
    tokenized_sentences: List[List[str]],
    max_n: int = 3,
) -> Dict[int, Counter]:
    """
    Build unigram, bigram, trigram frequency tables.

    Returns dict mapping n -> Counter of n-gram tuples.
    """
    tables = {n: Counter() for n in range(1, max_n + 1)}
    for tokens in tokenized_sentences:
        for n in range(1, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i : i + n])
                tables[n][ngram] += 1
    for n, counter in tables.items():
        logger.info(f"  {n}-gram types: {len(counter)}")
    return tables


def load_technical_vocabulary(tsv_path: str) -> List[Dict[str, str]]:
    """
    Load the hand-curated technical parallel corpus TSV.

    Returns list of dicts with columns as keys.
    """
    import csv

    records = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append(dict(row))
    logger.info(f"Loaded {len(records)} technical dictionary entries")
    return records


def assign_logit_bias(
    tech_vocab: List[Dict[str, str]],
    unigram_counts: Counter,
    bias_boost: float = 5.0,
) -> Dict[str, float]:
    """
    Assign logit bias weights to technical terms.

    Terms present in the technical dictionary but rare in the corpus
    get a positive bias to encourage their generation during decoding.

    Returns dict mapping Maithili term -> bias weight.
    """
    total_tokens = sum(unigram_counts.values()) or 1
    logit_bias = {}
    for entry in tech_vocab:
        term = entry.get("maithili", "").strip()
        if not term:
            continue
        freq = unigram_counts.get((term,), 0) / total_tokens
        if freq < 1e-5:
            logit_bias[term] = bias_boost
        elif freq < 1e-4:
            logit_bias[term] = bias_boost * 0.5
        else:
            logit_bias[term] = bias_boost * 0.1
    logger.info(f"Assigned logit bias to {len(logit_bias)} terms")
    return logit_bias


def build_kenlm_arpa(
    tokenized_sentences: List[List[str]],
    output_arpa_path: Path,
    order: int = 3,
):
    """
    Build an ARPA-format language model using KenLM's lmplz.

    Writes tokenized text to a temp file, then invokes lmplz.
    """
    import subprocess
    import tempfile

    text_path = output_arpa_path.parent / "corpus.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        for tokens in tokenized_sentences:
            f.write(" ".join(tokens) + "\n")

    logger.info(f"Building {order}-gram ARPA LM from {len(tokenized_sentences)} sentences...")
    try:
        result = subprocess.run(
            ["lmplz", "-o", str(order), "--text", str(text_path), "--arpa", str(output_arpa_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info(f"ARPA LM saved to {output_arpa_path}")
        else:
            logger.error(f"KenLM lmplz failed: {result.stderr}")
    except FileNotFoundError:
        logger.warning(
            "KenLM lmplz not found on PATH. "
            "Install KenLM or build the ARPA model manually."
        )


def export_logit_bias(logit_bias: Dict[str, float], output_path: Path):
    """Export logit bias dictionary as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(logit_bias, f, ensure_ascii=False, indent=2)
    logger.info(f"Exported logit bias ({len(logit_bias)} terms) to {output_path}")


def export_word_freq(ngram_tables: Dict[int, Counter], output_path: Path):
    """
    Export unigram word frequencies as JSON.

    Saves a dict mapping word -> count for all unigrams.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unigram = ngram_tables[1]
    freq_dict = {token[0]: count for token, count in unigram.most_common()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(freq_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Exported word_freq.json ({len(freq_dict)} words) to {output_path}")


def export_combined_vocab(
    ngram_tables: Dict[int, Counter],
    tech_terms: List[str],
    output_path: Path,
):
    """
    Export combined vocabulary (corpus unigrams + technical terms) as a text file,
    one word per line.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_words = {token[0] for token in ngram_tables[1]}
    tech_set = set(t.lower() for t in tech_terms)
    combined = sorted(corpus_words | tech_set)
    with open(output_path, "w", encoding="utf-8") as f:
        for word in combined:
            f.write(word + "\n")
    logger.info(f"Exported combined_vocab.txt ({len(combined)} words) to {output_path}")


def export_technical_vocab_json(
    tech_terms: List[str],
    unigram_counts: Counter,
    output_path: Path,
    base_bias: float = 5.0,
):
    """
    Export technical_vocab.json mapping each technical term to its logit bias weight.

    Bias assignment:
      - Terms absent from corpus or very rare (freq < 1e-5): base_bias (5.0)
      - Low frequency (1e-5 <= freq < 1e-4): base_bias * 0.6 (3.0)
      - Moderate frequency (1e-4 <= freq < 1e-3): base_bias * 0.4 (2.0)
      - Common (freq >= 1e-3): base_bias * 0.2 (minimal boost)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = sum(unigram_counts.values()) or 1

    vocab_with_bias = {}
    for term in tech_terms:
        term_lower = term.lower()
        freq = unigram_counts.get((term_lower,), 0) / total_tokens
        if freq < 1e-5:
            bias = base_bias
        elif freq < 1e-4:
            bias = round(base_bias * 0.6, 1)
        elif freq < 1e-3:
            bias = round(base_bias * 0.4, 1)
        else:
            bias = round(base_bias * 0.2, 1)
        vocab_with_bias[term] = bias

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_with_bias, f, ensure_ascii=False, indent=2)
    logger.info(
        f"Exported technical_vocab.json ({len(vocab_with_bias)} terms) to {output_path}"
    )


def prepare_ngram_data(config_path: str):
    """Main entry point: prepare N-gram LM and logit bias data."""
    config = load_config(config_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sangraha_dir = config["datasets"]["sangraha"]["save_dir"]
    tech_dict_path = config["technical_dictionary"]["output_path"]

    logger.info("Step 1: Loading Sangraha text...")
    texts = load_sangraha_text(sangraha_dir)

    logger.info("Step 2: Tokenizing with IndicNLP...")
    tokenized = tokenize_indicnlp(texts)

    logger.info("Step 3: Building N-gram frequency tables...")
    ngram_tables = build_ngram_tables(tokenized)

    logger.info("Step 4: Loading technical vocabulary...")
    if Path(tech_dict_path).exists():
        tech_vocab = load_technical_vocabulary(tech_dict_path)
    else:
        logger.warning(
            f"Technical dictionary TSV not found at {tech_dict_path}; "
            "using built-in TECHNICAL_TERMS list."
        )
        tech_vocab = [{"english": t, "maithili": t} for t in TECHNICAL_TERMS]

    logger.info("Step 5: Assigning logit bias weights...")
    logit_bias = assign_logit_bias(tech_vocab, ngram_tables[1])

    logger.info("Step 6: Building KenLM ARPA model...")
    build_kenlm_arpa(tokenized, OUTPUT_DIR / "maithili_3gram.arpa")

    logger.info("Step 7: Exporting logit bias dictionary...")
    export_logit_bias(logit_bias, OUTPUT_DIR / "logit_bias.json")

    logger.info("Step 8: Exporting word frequencies...")
    export_word_freq(ngram_tables, OUTPUT_DIR / "word_freq.json")

    logger.info("Step 9: Exporting combined vocabulary...")
    export_combined_vocab(ngram_tables, TECHNICAL_TERMS, OUTPUT_DIR / "combined_vocab.txt")

    logger.info("Step 10: Exporting technical vocabulary JSON...")
    export_technical_vocab_json(
        TECHNICAL_TERMS, ngram_tables[1], OUTPUT_DIR / "technical_vocab.json"
    )

    logger.info("N-gram data preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare N-gram LM data for Task 1.2"
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to dataset config YAML",
    )
    args = parser.parse_args()
    prepare_ngram_data(args.config)
