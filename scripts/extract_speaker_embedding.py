"""
Speaker embedding extraction for Speech Understanding PA-2, Task 3.1.

Extracts d-vector (Resemblyzer) or x-vector (SpeechBrain) from a voice recording
and saves the embedding as a .pt file for downstream voice cloning.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SpeakerEmbeddingExtractor:
    """Extract speaker embeddings via SpeechBrain x-vector or Resemblyzer d-vector."""

    def __init__(self, method: str = "speechbrain", device: str = None):
        self.method = method
        if device is None:
            from scripts.device_utils import get_device
            device = get_device()
        # SpeechBrain doesn't support MPS — fall back to CPU
        if device == "mps" and method == "speechbrain":
            device = "cpu"
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self):
        if self.method == "speechbrain":
            from speechbrain.inference.speaker import EncoderClassifier

            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                run_opts={"device": self.device},
            )
        elif self.method == "resemblyzer":
            from resemblyzer import VoiceEncoder

            self._model = VoiceEncoder(device=self.device)
        else:
            raise ValueError(f"Unknown method: {self.method!r}. Use 'speechbrain' or 'resemblyzer'.")

    def load_audio(self, audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """Load an audio file and resample to target_sr."""
        import librosa

        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio, target_sr

    def extract(self, audio_path: str) -> np.ndarray:
        """Extract a speaker embedding vector from an audio file.

        Returns:
            1-D numpy array with the embedding.
        """
        audio, sr = self.load_audio(audio_path, target_sr=16000)

        if self.method == "speechbrain":
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            embedding = (
                self._model.encode_batch(waveform)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
        elif self.method == "resemblyzer":
            from resemblyzer import preprocess_wav

            wav = preprocess_wav(audio, source_sr=sr)
            embedding = self._model.embed_utterance(wav)
        else:
            raise RuntimeError(f"Model not initialised for method {self.method!r}")

        logger.info(
            "Extracted %s embedding of shape %s from %s",
            self.method, embedding.shape, audio_path,
        )
        return embedding

    def save_embedding(self, embedding: np.ndarray, output_path: str) -> None:
        """Save an embedding as a .pt file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.tensor(embedding), str(out))
        logger.info("Saved embedding to %s", out)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Extract speaker embedding (Task 3.1)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="outputs/audio/student_voice_ref.wav",
        help="Path to input audio (e.g. 60s voice recording)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/audio/your_voice_embedding.pt",
        help="Path to save the embedding .pt file",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["speechbrain", "resemblyzer"],
        default="speechbrain",
        help="Embedding backend to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (auto-detects cuda/mps/cpu)",
    )
    args = parser.parse_args()

    extractor = SpeakerEmbeddingExtractor(method=args.method, device=args.device)
    embedding = extractor.extract(args.audio)
    extractor.save_embedding(embedding, args.output)
    print(f"Done. Embedding shape: {embedding.shape}  ->  {args.output}")


if __name__ == "__main__":
    main()
