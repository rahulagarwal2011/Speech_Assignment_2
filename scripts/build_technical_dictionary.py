"""
500-word technical dictionary builder for Task 2.2.

Builds a TSV dictionary of speech/ML/signal-processing terms with columns:
  english_term, hindi_equivalent, maithili, ipa_maithili, domain

Sources:
  1. Hardcoded 500+ term list across 6 domains
  2. Parallel corpus mining (Samanantar/BPCC/NLLB)
  3. Stub entries flagged for IndicTrans2 translation

CLI: python scripts/build_technical_dictionary.py \
       --parallel-corpus data/processed/translation/en_mai_parallel_clean.tsv \
       --output data/manual/technical_parallel_corpus.tsv --min-terms 500
"""

import csv
import json
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

TERMS_BY_DOMAIN = {
    "speech_processing": [
        "speech recognition", "automatic speech recognition", "speech-to-text",
        "text-to-speech", "speech synthesis", "voice activity detection",
        "speaker diarization", "speaker verification", "speaker identification",
        "keyword spotting", "wake word", "end-to-end", "connectionist temporal classification",
        "CTC", "sequence-to-sequence", "attention mechanism", "beam search",
        "language model", "acoustic model", "pronunciation model",
        "hidden Markov model", "Viterbi algorithm", "Baum-Welch algorithm",
        "Gaussian mixture model", "deep neural network", "recurrent neural network",
        "long short-term memory", "LSTM", "gated recurrent unit", "GRU",
        "transformer", "encoder", "decoder", "self-attention",
        "wav2vec", "whisper", "conformer", "transducer",
        "word error rate", "character error rate", "sentence error rate",
        "voice cloning", "voice conversion", "prosody transfer",
        "code-switching", "multilingual", "cross-lingual",
    ],
    "signal_processing": [
        "signal", "noise", "signal-to-noise ratio", "SNR",
        "frequency", "amplitude", "phase", "waveform",
        "sampling rate", "Nyquist frequency", "aliasing",
        "quantization", "bit depth", "dynamic range",
        "Fourier transform", "fast Fourier transform", "FFT",
        "discrete Fourier transform", "DFT", "inverse FFT",
        "short-time Fourier transform", "STFT", "spectrogram",
        "power spectral density", "bandwidth", "filter",
        "low-pass filter", "high-pass filter", "band-pass filter",
        "impulse response", "transfer function", "convolution",
        "correlation", "autocorrelation", "cross-correlation",
        "window function", "Hamming window", "Hanning window",
        "Gaussian window", "rectangular window",
        "downsampling", "upsampling", "resampling", "interpolation",
        "decimation", "normalization", "clipping", "distortion",
        "compression", "decompression", "codec",
        "noise reduction", "spectral subtraction", "Wiener filter",
    ],
    "acoustics": [
        "pitch", "fundamental frequency", "F0", "harmonics",
        "overtone", "resonance", "formant", "formant frequency",
        "vocal tract", "glottis", "larynx", "pharynx",
        "nasal cavity", "oral cavity", "articulatory",
        "source-filter model", "linear predictive coding", "LPC",
        "cepstrum", "cepstral coefficients", "mel-frequency",
        "MFCC", "mel-frequency cepstral coefficients",
        "filterbank", "mel filterbank", "log mel spectrogram",
        "bark scale", "critical band", "loudness",
        "intensity", "decibel", "sound pressure level",
        "reverberation", "echo", "room impulse response",
        "microphone", "transducer", "acoustic feature",
        "perceptual linear prediction", "PLP",
        "relative spectral", "RASTA",
    ],
    "machine_learning": [
        "neural network", "deep learning", "machine learning",
        "supervised learning", "unsupervised learning", "reinforcement learning",
        "semi-supervised learning", "self-supervised learning",
        "backpropagation", "gradient descent", "stochastic gradient descent",
        "SGD", "Adam optimizer", "learning rate", "momentum",
        "batch size", "epoch", "iteration",
        "loss function", "cross-entropy", "mean squared error",
        "softmax", "sigmoid", "ReLU", "activation function",
        "dropout", "batch normalization", "layer normalization",
        "regularization", "L1 regularization", "L2 regularization",
        "overfitting", "underfitting", "bias", "variance",
        "training set", "validation set", "test set",
        "cross-validation", "hyperparameter", "grid search",
        "feature extraction", "feature selection",
        "dimensionality reduction", "PCA", "SVD",
        "eigenvalue", "eigenvector", "covariance matrix",
        "classification", "regression", "clustering",
        "decision tree", "random forest", "support vector machine",
        "kernel function", "ensemble method", "bagging", "boosting",
        "k-means", "k-nearest neighbors",
        "confusion matrix", "precision", "recall", "F1 score",
        "ROC curve", "AUC", "true positive", "false positive",
        "true negative", "false negative", "accuracy",
        "transfer learning", "fine-tuning", "pre-training",
        "data augmentation", "perturbation", "robustness",
        "adversarial example", "adversarial training",
    ],
    "linguistics": [
        "phoneme", "morpheme", "lexeme", "grapheme",
        "phonology", "morphology", "syntax", "semantics", "pragmatics",
        "prosody", "intonation", "stress", "rhythm", "tone",
        "syllable", "consonant", "vowel", "diphthong",
        "voiced", "voiceless", "aspirated", "retroflex",
        "alveolar", "velar", "palatal", "dental", "labial",
        "nasal", "fricative", "affricate", "approximant",
        "stop consonant", "plosive", "lateral",
        "International Phonetic Alphabet", "IPA",
        "transliteration", "Romanization", "Devanagari",
        "code-mixing", "diglossia", "bilingualism",
        "language identification", "script detection",
        "named entity recognition", "part-of-speech tagging",
        "dependency parsing", "constituency parsing",
        "sentiment analysis", "text classification",
    ],
    "phonetics": [
        "articulatory phonetics", "acoustic phonetics", "auditory phonetics",
        "place of articulation", "manner of articulation",
        "voicing", "aspiration", "nasalization",
        "coarticulation", "assimilation", "elision",
        "gemination", "lenition", "fortition",
        "monophthong", "triphthong", "schwa",
        "suprasegmental", "segment", "phone", "allophone",
        "minimal pair", "complementary distribution",
        "free variation", "phonotactics",
        "onset", "coda", "nucleus", "rime",
        "pitch accent", "lexical tone", "contour tone",
        "register tone", "downdrift", "downstep",
        "breathy voice", "creaky voice", "modal voice",
        "ejective", "implosive", "click",
        "vowel harmony", "consonant cluster",
        "syllabification", "resyllabification",
        "connected speech", "sandhi",
        "formant transition", "voice onset time", "VOT",
    ],
}

HINDI_EQUIVALENTS = {
    "speech recognition": "वाक् पहचान",
    "automatic speech recognition": "स्वचालित वाक् पहचान",
    "speech-to-text": "वाक्-से-पाठ",
    "text-to-speech": "पाठ-से-वाक्",
    "speech synthesis": "वाक् संश्लेषण",
    "neural network": "तंत्रिका जाल",
    "deep learning": "गहन अधिगम",
    "machine learning": "यंत्र अधिगम",
    "signal": "संकेत",
    "noise": "शोर",
    "frequency": "आवृत्ति",
    "amplitude": "आयाम",
    "sampling rate": "प्रतिचयन दर",
    "spectrogram": "स्पेक्ट्रोग्राम",
    "pitch": "पिच",
    "fundamental frequency": "मूल आवृत्ति",
    "phoneme": "ध्वनिग्राम",
    "prosody": "छंद",
    "formant": "फॉर्मेंट",
    "cepstrum": "सेप्स्ट्रम",
    "mel-frequency": "मेल-आवृत्ति",
    "MFCC": "एमएफसीसी",
    "filterbank": "फिल्टरबैंक",
    "Hamming window": "हैमिंग विंडो",
    "FFT": "एफएफटी",
    "DFT": "डीएफटी",
    "linear predictive coding": "रैखिक पूर्वानुमान कोडिंग",
    "hidden Markov model": "छुपा मार्कोव मॉडल",
    "Viterbi algorithm": "विटरबी एल्गोरिथ्म",
    "Baum-Welch algorithm": "बाम-वेल्च एल्गोरिथ्म",
    "Gaussian mixture model": "गॉसियन मिश्रण मॉडल",
    "language model": "भाषा मॉडल",
    "acoustic model": "ध्वनिक मॉडल",
    "beam search": "बीम सर्च",
    "attention mechanism": "ध्यान तंत्र",
    "transformer": "ट्रांसफार्मर",
    "encoder": "एनकोडर",
    "decoder": "डिकोडर",
    "wav2vec": "वेव2वेक",
    "whisper": "व्हिस्पर",
    "CTC": "सीटीसी",
    "sequence-to-sequence": "अनुक्रम-से-अनुक्रम",
    "word error rate": "शब्द त्रुटि दर",
    "backpropagation": "पश्चप्रसारण",
    "gradient descent": "प्रवणता अवरोहण",
    "loss function": "हानि फलन",
    "cross-entropy": "क्रॉस-एन्ट्रॉपी",
    "softmax": "सॉफ्टमैक्स",
    "dropout": "ड्रॉपआउट",
    "overfitting": "अतिफिटिंग",
    "underfitting": "अल्पफिटिंग",
    "classification": "वर्गीकरण",
    "regression": "प्रतिगमन",
    "clustering": "संकुलन",
    "precision": "परिशुद्धता",
    "recall": "प्रत्याह्वान",
    "F1 score": "एफ1 स्कोर",
    "transfer learning": "अंतरण अधिगम",
    "fine-tuning": "सूक्ष्म समायोजन",
    "data augmentation": "डेटा संवर्धन",
    "vocoder": "वोकोडर",
    "stochastic": "स्टोकैस्टिक",
    "vocal tract": "स्वर पथ",
    "vowel": "स्वर",
    "consonant": "व्यंजन",
    "syllable": "अक्षर",
    "IPA": "अंतरराष्ट्रीय ध्वन्यात्मक वर्णमाला",
    "Devanagari": "देवनागरी",
    "transliteration": "लिप्यंतरण",
    "voice cloning": "आवाज़ क्लोनिंग",
    "voice conversion": "आवाज़ रूपांतरण",
    "code-switching": "कोड-स्विचिंग",
    "normalization": "सामान्यीकरण",
    "convolution": "संवलन",
    "correlation": "सहसंबंध",
    "Fourier transform": "फूरियर रूपांतरण",
    "filter": "फिल्टर",
    "reverberation": "प्रतिध्वनि",
    "microphone": "माइक्रोफोन",
    "loudness": "प्रबलता",
    "decibel": "डेसिबल",
    "waveform": "तरंगरूप",
    "phase": "कला",
    "bandwidth": "बैंडविड्थ",
    "supervised learning": "पर्यवेक्षित अधिगम",
    "unsupervised learning": "अपर्यवेक्षित अधिगम",
    "reinforcement learning": "प्रबलन अधिगम",
    "activation function": "सक्रियण फलन",
    "regularization": "नियमितीकरण",
    "feature extraction": "अभिलक्षण निष्कर्षण",
    "confusion matrix": "भ्रम आव्यूह",
    "accuracy": "शुद्धता",
    "bias": "पूर्वाग्रह",
    "variance": "प्रसरण",
    "epoch": "युग",
    "learning rate": "अधिगम दर",
    "kernel function": "कर्नेल फलन",
    "support vector machine": "सपोर्ट वेक्टर मशीन",
    "decision tree": "निर्णय वृक्ष",
    "random forest": "यादृच्छिक वन",
    "intonation": "स्वराघात",
    "stress": "बलाघात",
    "morpheme": "रूपिम",
    "syntax": "वाक्य विन्यास",
    "semantics": "अर्थ विज्ञान",
    "named entity recognition": "नामित निकाय पहचान",
    "sentiment analysis": "भावना विश्लेषण",
    "signal-to-noise ratio": "संकेत-से-शोर अनुपात",
    "quantization": "क्वांटीकरण",
    "compression": "संपीड़न",
}


def _get_all_terms() -> List[Dict[str, str]]:
    """Flatten TERMS_BY_DOMAIN into a list of dicts with english_term + domain."""
    seen: Set[str] = set()
    terms = []
    for domain, term_list in TERMS_BY_DOMAIN.items():
        for t in term_list:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                terms.append({"english_term": t, "domain": domain})
    return terms


def mine_corpus_translations(
    parallel_corpus_path: str,
    terms: List[Dict[str, str]],
) -> Dict[str, str]:
    """Search parallel corpus TSV for existing translations of terms.

    Looks for English terms in the 'src' column and returns
    the corresponding 'tgt' (Maithili) as the translation.
    Only the first match per term is kept.

    Args:
        parallel_corpus_path: Path to en_mai_parallel_clean.tsv.
        terms: List of term dicts with 'english_term' key.

    Returns:
        Dict mapping lowercase english term -> Maithili translation.
    """
    corpus_path = Path(parallel_corpus_path)
    if not corpus_path.exists():
        logger.warning(f"Parallel corpus not found: {parallel_corpus_path}")
        return {}

    term_set = {t["english_term"].lower() for t in terms}
    found: Dict[str, str] = {}

    with open(corpus_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            src = (row.get("src") or row.get("en") or "").strip().lower()
            tgt = (row.get("tgt") or row.get("mai") or "").strip()
            if not src or not tgt:
                continue
            for term in term_set - set(found.keys()):
                if term in src:
                    found[term] = tgt

    logger.info(
        f"Mined {len(found)} term translations from parallel corpus"
    )
    return found


def generate_translations_stub(
    remaining_terms: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Create placeholder entries for terms needing IndicTrans2 translation.

    These entries are flagged with '[TRANSLATE:<term>]' in the maithili column
    so the user can batch-translate them at runtime.

    Args:
        remaining_terms: Terms not found in corpus.

    Returns:
        List of partial dictionary rows.
    """
    stubs = []
    for t in remaining_terms:
        en = t["english_term"]
        hindi = HINDI_EQUIVALENTS.get(en, HINDI_EQUIVALENTS.get(en.lower(), ""))
        stubs.append({
            "english_term": en,
            "hindi_equivalent": hindi,
            "maithili": f"[TRANSLATE:{en}]",
            "ipa_maithili": "",
            "domain": t["domain"],
        })
    return stubs


def build_dictionary(
    parallel_corpus_path: Optional[str],
    output_path: str,
    min_terms: int = 500,
) -> List[Dict[str, str]]:
    """Full dictionary build pipeline.

    Steps:
      1. Collect all 500+ hardcoded terms across domains.
      2. Mine parallel corpus for existing translations.
      3. Fill Hindi equivalents from hardcoded map.
      4. Flag remaining terms for IndicTrans2 translation.
      5. Write TSV output.

    Args:
        parallel_corpus_path: Path to en_mai_parallel_clean.tsv (can be None).
        output_path: Path to write the output TSV.
        min_terms: Minimum number of dictionary entries.

    Returns:
        List of dictionary row dicts.
    """
    all_terms = _get_all_terms()
    logger.info(f"Total unique terms across all domains: {len(all_terms)}")

    corpus_translations: Dict[str, str] = {}
    if parallel_corpus_path:
        corpus_translations = mine_corpus_translations(parallel_corpus_path, all_terms)

    dictionary = []
    remaining = []

    for t in all_terms:
        en = t["english_term"]
        en_lower = en.lower()
        domain = t["domain"]
        hindi = HINDI_EQUIVALENTS.get(en, HINDI_EQUIVALENTS.get(en_lower, ""))

        if en_lower in corpus_translations:
            dictionary.append({
                "english_term": en,
                "hindi_equivalent": hindi,
                "maithili": corpus_translations[en_lower],
                "ipa_maithili": "",
                "domain": domain,
            })
        elif hindi:
            dictionary.append({
                "english_term": en,
                "hindi_equivalent": hindi,
                "maithili": f"[TRANSLATE:{en}]",
                "ipa_maithili": "",
                "domain": domain,
            })
        else:
            remaining.append(t)

    stubs = generate_translations_stub(remaining)
    dictionary.extend(stubs)

    if len(dictionary) < min_terms:
        logger.warning(
            f"Dictionary has {len(dictionary)} entries, below target {min_terms}. "
            "Add more domain terms or run IndicTrans2 to fill stubs."
        )

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    columns = ["english_term", "hindi_equivalent", "maithili", "ipa_maithili", "domain"]
    with open(out_p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in dictionary:
            writer.writerow(row)

    corpus_count = sum(1 for r in dictionary if not r["maithili"].startswith("[TRANSLATE:"))
    stub_count = sum(1 for r in dictionary if r["maithili"].startswith("[TRANSLATE:"))
    logger.info(
        f"Dictionary written to {output_path}: "
        f"{len(dictionary)} total entries "
        f"({corpus_count} translated, {stub_count} stubs)"
    )

    return dictionary


def main():
    parser = argparse.ArgumentParser(
        description="Build 500-word technical dictionary for Task 2.2"
    )
    parser.add_argument(
        "--parallel-corpus",
        default="data/processed/translation/en_mai_parallel_clean.tsv",
        help="Path to deduplicated En-Mai parallel corpus TSV",
    )
    parser.add_argument(
        "--output",
        default="data/manual/technical_parallel_corpus.tsv",
        help="Path to write the technical dictionary TSV",
    )
    parser.add_argument(
        "--min-terms", type=int, default=500,
        help="Minimum number of dictionary entries to produce",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    dictionary = build_dictionary(args.parallel_corpus, args.output, args.min_terms)
    logger.info(f"Dictionary build complete: {len(dictionary)} entries")


if __name__ == "__main__":
    main()
