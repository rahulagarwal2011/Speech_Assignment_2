# Implementation Note — Speech Understanding PA-2

> **Rahul Agarwal | Roll No. M25DE1040**
> One non-obvious design choice per pipeline part.

---

## Part I: Script-Based Dual-Pass Language Detection for Code-Switched Audio

**Design choice:** Whisper's `detect_language()` consistently misclassifies Hindi-dominant code-switched audio as English because the acoustic features of Hinglish overlap heavily with English. Rather than relying on this detector, we implemented a **script-based dual-pass approach**: for each 10-second window, the audio is transcribed as Hindi (`language="hi"`, `task="transcribe"`), and the output text is analyzed for Devanagari characters (Unicode range `\u0900-\u097F`). If >30% of characters are Devanagari, the window is labeled Hindi; otherwise English.

This solved the critical problem where Whisper was *translating* Hindi speech into English text (e.g., "दुनिया की बड़ी यूनिवर्सिटीज" became "The world's big universities should do its case study"). With script detection, the pipeline correctly identified **86 Hindi segments** and **2 English segments**, producing proper Devanagari transcription for Hindi portions. The frame-level CNN+BiGRU LID model (F1=1.0 for En/Hi on validation) was trained on Whisper-bootstrapped labels from the same lecture audio, providing a second layer of language boundary detection.

---

## Part II: 8-Domain Dictionary with Stub-Based Deferred Translation

**Design choice:** Rather than building a 500-term dictionary from a single domain, we spread **511 terms** across 8 specialized domains: speech processing, signal processing, ML, acoustics, linguistics, phonetics, anti-spoofing, and translation/NMT. Terms without verified Maithili translations are stored as `[TRANSLATE:<term>]` stubs, enabling batch translation via IndicTrans2 at runtime without blocking the pipeline.

The 8-domain spread ensures coverage of not just obvious terms ("spectrogram") but also assignment-relevant terminology ("equal error rate", "adversarial perturbation", "byte pair encoding"). Hindi equivalents were hand-curated for 100+ high-frequency terms, making the dictionary genuinely useful for translating the full lecture content.

---

## Part III: Dual-Model TTS with MCD-Based Automatic Selection

**Design choice:** The pipeline synthesizes with **both** XTTS v2 (zero-shot voice cloning) and MMS-TTS (language-specific), then automatically selects the model with lower Mel-Cepstral Distortion against the student reference. In our run, both models succeeded:

| Model | MCD | Voice Cloning |
|-------|-----|---------------|
| XTTS v2 | 14.70 dB | Yes (uses student voice) |
| MMS-TTS | **10.19 dB** | No (fixed Maithili speaker) |

MMS was selected for lower MCD despite lacking voice cloning. This highlights a trade-off: XTTS preserves the student's voice identity (critical for the assignment's "zero-shot cloning" requirement) but produces slightly higher distortion due to cross-lingual voice adaptation. The architecture supports overriding the selection to use XTTS when voice identity preservation is prioritized over raw MCD. Loading XTTS required patching PyTorch 2.6's `weights_only=True` default via `torch.serialization.add_safe_globals` and a `torch.load` monkey-patch fallback — a non-trivial compatibility fix for the TTS library.

---

## Part IV: Binary Search for Minimum Imperceptible $\varepsilon$

**Design choice:** Instead of testing fixed $\varepsilon$ values, we implement a **binary search over 20 iterations** ($[10^{-4}, 0.1]$) to find the minimum FGSM perturbation that flips the LID prediction while maintaining SNR > 40 dB.

Result: even $\varepsilon = 1.0 \times 10^{-4}$ achieved a flip at SNR = 61.5 dB — well above the 40 dB imperceptibility threshold. The complete tradeoff curve shows monotonic SNR improvement as epsilon decreases:

| $\varepsilon$ | SNR (dB) | Flipped |
|---|---|---|
| 0.050 | 7.5 | Yes |
| 0.006 | 25.4 | Yes |
| 0.0009 | 42.6 | Yes |
| 0.0001 | 61.5 | Yes |

This extreme sensitivity — flipping at the smallest tested epsilon — indicates the LID model's decision boundary is very close to the input manifold, expected for a model trained on bootstrapped pseudo-labels without adversarial training. The 20-point curve provides evidence that FGSM attacks are predictable and that perturbations at $\varepsilon \leq 10^{-3}$ are truly inaudible (SNR > 40 dB).

---

*End of implementation note.*
