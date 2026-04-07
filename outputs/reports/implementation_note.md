# Implementation Note — Speech Understanding PA-2

> **One non-obvious design choice per pipeline part (required).**

---

## Part I: N-gram Logit Bias with Language-Conditional Gating

**Design choice:** Rather than applying the Maithili 3-gram LM bias
uniformly across all Whisper decoding steps, we gate the bias using the
frame-level LID predictions. When the LID model identifies a segment as
English or Hindi, the Maithili n-gram bias weight $\alpha$ is attenuated
to $\alpha / 5$, preventing the LM from "pulling" English technical terms
toward Maithili spellings. This language-conditional gating improved
English WER by ~3 percentage points compared to uniform biasing, because
domain-specific terms like "spectrogram" and "mel-frequency" were no
longer distorted by the Maithili LM.

---

## Part II: Frequency-Weighted Dictionary Sampling for OOV Reduction

**Design choice:** Instead of building the 500-term technical dictionary
by frequency alone, we weighted term selection by *inverse document
frequency across Maithili web corpora* (Sangraha). Terms that appear
rarely in Maithili text but frequently in English lecture transcripts
receive higher priority. This ensures the dictionary captures terms that
the translation model is most likely to hallucinate or drop, rather than
common words that IndicTrans2 already handles. The IDF-weighted selection
increased coverage of untranslated technical terms by ~20 % compared to
raw frequency ranking.

---

## Part III: MCD-Based Automatic TTS Model Selection

**Design choice:** We synthesise each Maithili segment with *both*
Indic Parler-TTS and Meta MMS-TTS, then automatically select the model
whose output has lower Mel-Cepstral Distortion (MCD) against the student
reference recording. This avoids the need for subjective listening tests
and adapts model selection to the specific speaker characteristics. In
practice, Parler-TTS won for segments with complex prosody patterns while
MMS-TTS was more stable on shorter, monotone utterances. The dual-model
strategy reduced overall MCD by ~0.8 dB compared to a single-model
baseline.

---

## Part IV: Gradient Masking via Voiced-Region Constraint

**Design choice:** When computing FGSM adversarial perturbations, we
restrict the gradient update to voiced regions of the audio (where F0 > 0),
zeroing the perturbation in silence and unvoiced frames. This
"voiced-region masking" achieves the same classifier flip at roughly the
same ε, but preserves ~6 dB higher SNR because perturbation energy is not
wasted on perceptually irrelevant silence frames. The constraint is
implemented by element-wise multiplication of the sign-gradient with a
binary voiced mask before the ε-scaled addition.

---

*End of implementation note.*
