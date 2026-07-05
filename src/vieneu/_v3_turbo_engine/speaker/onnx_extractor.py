"""ONNX speaker embedding extractor.

The encoder is a frozen ONNX graph: it takes an 80-dim Kaldi fbank
(mean-normalized, 16 kHz) → a 192-d global speaker embedding. IO signature:
    input  "input"  : (batch, seq_len, 80) float
    output "output" : (batch, 192)         float

Used both offline (scripts/extract_xvector.py) to cache per-utterance embeddings, and
at inference (inference.py) to extract one vector per reference voice. Only the small
trainable `xvec_proj` inside the TTS model is learned; this encoder is never trained.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import numpy as np
import torch

from .fbank import _SPEAKER_FBANK_SAMPLE_RATE, extract_speaker_fbank

EMBED_DIM = 192  # speaker embedding output dimension


class OnnxSpeakerEncoder:
    """Frozen speaker encoder backed by onnxruntime."""

    def __init__(
        self,
        onnx_path: str,
        device: str = "cpu",
        max_seconds: float = 30.0,
        providers: Optional[Sequence[str]] = None,
    ):
        import onnxruntime as ort

        if providers is None:
            avail = set(ort.get_available_providers())
            if "cuda" in str(device).lower() and "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name    # "input"
        self.output_name = self.session.get_outputs()[0].name  # "output"
        self.max_seconds = float(max_seconds)
        self.embedding_dim = EMBED_DIM
        self.onnx_path = onnx_path

    # ── Loading ─────────────────────────────────────────────────────────────
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: Union[str, tuple, list],
        filename: str = "speaker_encoder.onnx",
        device: str = "cpu",
        **kwargs,
    ) -> "OnnxSpeakerEncoder":
        """Resolve `speaker_encoder.onnx` from a local file, a local dir, or an HF repo id.

        Accepts a (repo_or_dir, filename) tuple too. When `path_or_repo` is neither a
        local file nor dir it is treated as a Hugging Face repo and the file is fetched
        via hf_hub_download — so a model whose checkpoint repo bundles speaker_encoder.onnx
        works out of the box.
        """
        if isinstance(path_or_repo, (tuple, list)):
            path_or_repo, filename = path_or_repo[0], path_or_repo[1]
        if os.path.isfile(path_or_repo):
            onnx_path = path_or_repo
        elif os.path.isdir(path_or_repo):
            onnx_path = os.path.join(path_or_repo, filename)
            if not os.path.isfile(onnx_path):
                raise FileNotFoundError(f"{filename} not found in directory {path_or_repo}")
        else:
            from huggingface_hub import hf_hub_download
            onnx_path = hf_hub_download(path_or_repo, filename)
        return cls(onnx_path, device=device, **kwargs)

    # ── Extraction ──────────────────────────────────────────────────────────
    def _to_mono(self, wav: torch.Tensor) -> torch.Tensor:
        wav = torch.as_tensor(wav, dtype=torch.float32)
        if wav.ndim == 2:
            wav = wav.mean(0) if wav.shape[0] > 1 else wav[0]
        elif wav.ndim != 1:
            raise ValueError(f"Expected 1D or 2D waveform, got shape {tuple(wav.shape)}")
        return wav

    @torch.no_grad()
    def embed(self, wav, sr: int) -> np.ndarray:
        """One waveform → (192,) float32 x-vector.

        `wav` may be a (T,) or (ch, T) tensor/array; it is downmixed to mono, capped to
        `max_seconds`, turned into an 80-dim mean-normed fbank (resampled to 16 kHz by
        extract_speaker_fbank), and run through the ONNX graph.
        """
        mono = self._to_mono(wav)
        if self.max_seconds > 0:
            cap = int(sr * self.max_seconds)
            if mono.numel() > cap:
                mono = mono[:cap]
        feat = extract_speaker_fbank(mono, sample_rate=sr)          # (T, 80) @ 16 kHz, mean-normed
        feat = feat.unsqueeze(0).cpu().numpy().astype(np.float32)   # (1, T, 80)
        out = self.session.run([self.output_name], {self.input_name: feat})[0]  # (1, 192)
        return out[0].astype(np.float32)

    @torch.no_grad()
    def embed_fbank(self, fbank: torch.Tensor) -> np.ndarray:
        """Precomputed (T, 80) mean-normed fbank → (192,) x-vector (skips audio I/O)."""
        feat = torch.as_tensor(fbank, dtype=torch.float32).unsqueeze(0).cpu().numpy()
        out = self.session.run([self.output_name], {self.input_name: feat})[0]
        return out[0].astype(np.float32)

    # Allow calling like the old torch module: enc(wav, sr)
    def __call__(self, wav, sr: int = _SPEAKER_FBANK_SAMPLE_RATE) -> np.ndarray:
        return self.embed(wav, sr)
