"""
VieNeu-TTS v3 Turbo — torch-free ONNX engine (CPU).
===================================================
A PyTorch-free inference engine for the ``update`` architecture (speaker-embedding
conditioning + single-layer acoustic decoder). The transformer forwards and the
MOSS audio codec run in ONNX Runtime; everything else (embeddings, the speaker
anchor, output heads, sampling, prompt build) is plain NumPy.

Synthesis from a preset / precomputed voice (``speaker_emb`` + ``ref_codes``) is
fully torch-free. Cloning a fresh reference wav (:meth:`prepare_reference`) also
needs a denoiser + speaker encoder: the denoiser is torch-free (numpy + ORT); the
speaker encoder's fbank front-end uses torchaudio, imported lazily only then.

Artifacts (fetched from HF ``<repo>/<onnx_subfolder>``, or a local dir):
  graphs : vieneu_prefill.onnx, vieneu_decode_step.onnx, vieneu_acoustic_cached.onnx
           + vieneu_backbone_shared.data + vieneu_v3_heads.npz (tied emb/heads + xvec_proj)
  setup  : config.json, tokenizer.json (bundled next to the graphs)
  codec  : <codec_repo>/moss_audio_tokenizer_{decode_full,encode}.onnx (+ .data)
  cloning: <repo>/speaker_encoder.onnx + <repo>/denoiser.onnx (repo root)

The public API mirrors ``VieNeuTTSv3Turbo`` (prepare_reference / infer) so
``V3TurboVieNeuTTS`` can use it as a drop-in CPU engine.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple, Union

import numpy as np

_V3_REPO = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
_GRAPH_FILES = [
    "vieneu_prefill.onnx", "vieneu_decode_step.onnx",
    "vieneu_acoustic_cached.onnx", "vieneu_backbone_shared.data",
    "vieneu_v3_heads.npz", "config.json", "tokenizer.json",
]
_CODEC_FILES = [
    "moss_audio_tokenizer_decode_full.onnx", "moss_audio_tokenizer_decode_shared.data",
    "moss_audio_tokenizer_decode_step.onnx", "codec_browser_onnx_meta.json",
    "moss_audio_tokenizer_encode.onnx", "moss_audio_tokenizer_encode.data",
]
_MAX_REF_SECONDS = 8.0
_STREAM_LEADIN_FRAMES = 4


class _Dev:
    """Minimal stand-in so callers' ``engine.device.type == 'cuda'`` works."""
    type = "cpu"


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class OnnxV3LiteEngine:
    SAMPLE_RATE = 48_000

    def __init__(
        self,
        checkpoint_path: str = _V3_REPO,
        onnx_repo: Optional[str] = None,
        codec_repo: str = _CODEC_REPO,
        onnx_dir: Optional[str] = None,
        onnx_subfolder: str = "onnx_int8",   # int8 backbone (mặc định); "onnx_update" = fp32
        codec_dir: Optional[str] = None,
        threads: int = 0,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        **_kw,
    ):
        import onnxruntime as ort

        self._lock = threading.RLock()
        self.device = _Dev()
        self.checkpoint_path = checkpoint_path
        repo = onnx_repo or checkpoint_path

        if not onnx_dir:
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(checkpoint_path, "config.json")
            except Exception:
                pass

        # ── Fetch artifacts (graphs + config + tokenizer, all in onnx_subfolder) ──
        if onnx_dir:
            vd = Path(onnx_dir)
        else:
            vd = self._fetch(repo, _GRAPH_FILES, onnx_subfolder)
        cd = Path(codec_dir) if codec_dir else self._fetch(codec_repo, _CODEC_FILES, None)

        # ── Config (token ids, arch sizes) ─────────────────────────────────────
        c = json.loads((vd / "config.json").read_text(encoding="utf-8"))
        self.cfg = c
        self.n_vq = int(c["n_vq"])
        self.hidden = int(c["hidden_size"])
        self.L = int(c["num_hidden_layers"])
        self.L_loc = int(c.get("local_num_hidden_layers", 1))
        self.nH_loc = int(c.get("local_num_attention_heads", 8))
        self.hd_loc = self.hidden // self.nH_loc
        self.audio_pad = int(c["audio_pad_token_id"])
        self.tps = int(c["text_prompt_start_token_id"])
        self.tpe = int(c["text_prompt_end_token_id"])
        self.sgs = int(c["speech_generation_start_token_id"])
        self.eos_speech = int(c["speech_generation_end_token_id"])
        self.ref_slot = int(c["audio_ref_slot_token_id"])
        self.text_vocab = int(c["text_vocab_size"])
        self.default_style_id = int(c.get("default_style_token_id", 16))
        self.style_labels = dict(c.get("style_labels", {}))
        self.use_speaker_embedding = bool(c.get("use_speaker_embedding", False))
        self.speaker_embedding_dim = int(c.get("speaker_embedding_dim", 192))
        self.speaker_encoder_filename = c.get("speaker_encoder_filename", "speaker_encoder.onnx")

        # ── Tied embeddings/heads + xvec_proj (numpy) ──────────────────────────
        z = np.load(vd / "vieneu_v3_heads.npz")
        self.text_emb = z["text_emb"].astype(np.float32)            # (Vt, H)
        self.audio_emb = z["audio_emb"].astype(np.float32)          # (n_vq, Va, H)
        if self.use_speaker_embedding and "xvec_w" in z:
            self.xvec_w = z["xvec_w"].astype(np.float32)            # (H, spk_dim)
            self.xvec_b = z["xvec_b"].astype(np.float32)            # (H,)
            self.xvec_ln_w = z["xvec_ln_w"].astype(np.float32)      # (H,)
            self.xvec_ln_b = z["xvec_ln_b"].astype(np.float32)      # (H,)
            self.xvec_ln_eps = float(z["xvec_ln_eps"])
        else:
            self.xvec_w = None

        # ── Tokenizer (tokenizers lib, torch-free) ─────────────────────────────
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(vd / "tokenizer.json"))

        # ── ONNX sessions ──────────────────────────────────────────────────────
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if threads and threads > 0:
            intra = int(threads)
        else:
            intra = min(max((os.cpu_count() or 8) // 2, 1), 8)
        so.intra_op_num_threads = intra
        self.ort_intra_op_threads = intra
        self.onnx_providers = list(providers)
        prov = self.onnx_providers
        self.sess_pre = ort.InferenceSession(str(vd / "vieneu_prefill.onnx"), so, providers=prov)
        self.sess_dec = ort.InferenceSession(str(vd / "vieneu_decode_step.onnx"), so, providers=prov)
        self.sess_ac = ort.InferenceSession(str(vd / "vieneu_acoustic_cached.onnx"), so, providers=prov)
        self.sess_codec_dec = ort.InferenceSession(
            str(cd / "moss_audio_tokenizer_decode_full.onnx"), so, providers=prov)
        self._codec_enc_path = str(cd / "moss_audio_tokenizer_encode.onnx")
        self._sess_codec_enc = None  # lazy (only for cloning)
        # Streaming codec decoder (decode_step) for infer_stream. Bit-exact to the full
        # decoder when driven with the documented state (cached_positions init = -1).
        self.sess_codec_step = None
        self._codec_stream_spec = None
        try:
            meta = json.loads((cd / "codec_browser_onnx_meta.json").read_text(encoding="utf-8"))
            self._codec_stream_spec = meta.get("streaming_decode")
            self.sess_codec_step = ort.InferenceSession(
                str(cd / "moss_audio_tokenizer_decode_step.onnx"), so, providers=prov)
            self._codec_step_out_names = [o.name for o in self.sess_codec_step.get_outputs()]
        except Exception:
            self.sess_codec_step = None  # streaming decode unavailable → infer_stream falls back

        # ── Speaker encoder + denoiser (voice cloning), from repo root ──────────
        # Loaded lazily on first clone: the denoiser is torch-free, the speaker
        # encoder's fbank front-end pulls in torchaudio.
        self.speaker_encoder = None
        self.denoiser = self._load_denoiser()

    # ── artifact helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _fetch(repo: str, files: List[str], subfolder: Optional[str]) -> Path:
        from huggingface_hub import hf_hub_download
        last = None
        for fn in files:
            try:
                last = hf_hub_download(repo, fn, repo_type="model", subfolder=subfolder or None)
            except Exception:
                if fn.endswith((".json",)):
                    continue  # optional meta may live elsewhere
                raise
        return Path(last).parent

    def _load_denoiser(self):
        try:
            from .onnx_denoiser import OnnxDenoiser
            path = self._resolve_root_file("denoiser.onnx")
            return OnnxDenoiser(path, providers=self.onnx_providers) if path else None
        except Exception:
            return None

    def _resolve_root_file(self, filename: str) -> Optional[str]:
        """Find a repo-root artifact (speaker_encoder.onnx / denoiser.onnx)."""
        p = Path(self.checkpoint_path)
        if p.is_dir() and (p / filename).is_file():
            return str(p / filename)
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(self.checkpoint_path, filename)
        except Exception:
            return None

    def _ensure_speaker_encoder(self):
        if self.speaker_encoder is None:
            from .speaker import OnnxSpeakerEncoder
            self.speaker_encoder = OnnxSpeakerEncoder.from_pretrained(
                self.checkpoint_path,
                filename=self.speaker_encoder_filename,
                device="cpu",
                providers=self.onnx_providers,
            )
        return self.speaker_encoder

    # ── numpy embedding / speaker anchor / heads / sampling ────────────────────
    def _speaker_anchor(self, speaker_emb: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """192-d x-vector → (H,) anchor (mirror xvec_proj: Linear + LayerNorm)."""
        if not self.use_speaker_embedding:
            return None
        if speaker_emb is None:
            raise ValueError("This model needs a speaker anchor: pass `speaker_emb` (see prepare_reference).")
        if self.xvec_w is None:
            raise RuntimeError("heads.npz has no xvec_proj weights — re-export with the update model.")
        v = np.asarray(speaker_emb, dtype=np.float32).reshape(-1)
        if not v.any():
            raise ValueError("speaker_emb is all-zero — not a valid speaker anchor.")
        v = v @ self.xvec_w.T + self.xvec_b                         # (H,)
        v = (v - v.mean()) / np.sqrt(v.var() + self.xvec_ln_eps)
        return (v * self.xvec_ln_w + self.xvec_ln_b).astype(np.float32)

    def _embed_rows(self, rows: np.ndarray, anchor: Optional[np.ndarray] = None) -> np.ndarray:
        """rows: (T, n_vq+1) int → (1, T, H) float32 (mirror _build_inputs_embeds)."""
        emb = self.text_emb[rows[:, 0]]                             # (T, H)
        for ch in range(self.n_vq):
            ids = rows[:, ch + 1]
            valid = ids != self.audio_pad
            safe = np.where(valid, ids, 0)
            emb = emb + self.audio_emb[ch][safe] * valid[:, None]
        if anchor is not None:
            emb = emb + anchor[None]
        return emb[None].astype(np.float32)

    def _sample(self, logits, temperature, top_k, top_p, rep_pen, prev):
        logits = logits.astype(np.float32)
        if not math.isclose(rep_pen, 1.0) and prev:
            idx = np.fromiter(prev, dtype=np.int64, count=len(prev))
            sel = logits[idx]
            logits = logits.copy()
            logits[idx] = np.where(sel < 0, sel * rep_pen, sel / rep_pen)
        if not (temperature and temperature > 0):
            return int(logits.argmax())
        logits = logits / temperature
        V = logits.shape[-1]
        # top_k FIRST → sort/softmax/choice run over just the k candidates, not the full
        # vocab. Identical distribution, but drops a full-vocab argsort+softmax each call —
        # ~1.5x faster sampling (≈1/3 of per-frame CPU time on the lite ONNX path).
        if top_k and 0 < int(top_k) < V:
            k = int(top_k)
            cand = np.argpartition(logits, -k)[-k:]            # k chỉ số lớn nhất (chưa sort)
        else:
            cand = np.arange(V)
        cs = logits[cand]
        order = np.argsort(cs)[::-1]                            # sort CHỈ trên k ứng viên
        cand = cand[order]
        p = _softmax(cs[order])
        if top_p and top_p < 1.0:
            keep = (np.cumsum(p) - p) < top_p                   # nucleus trong tập ứng viên
            p = p * keep
            p = p / p.sum()
        return int(cand[np.random.choice(cand.shape[-1], p=p)])

    # ── style / prompt build (numpy, mirror build_prompt_2d) ───────────────────
    def _resolve_style_id(self, style) -> int:
        if isinstance(style, (int, np.integer)):
            return int(style)
        return int(self.style_labels.get(style, self.default_style_id))

    def _build_rows(self, phonemes: str, ref_codes: Optional[np.ndarray], style_id: int) -> np.ndarray:
        phone_ids = self.tokenizer.encode(phonemes, add_special_tokens=False).ids
        text_ids = [style_id, self.tps] + list(phone_ids) + [self.tpe]
        T = len(text_ids)
        rows = np.full((T, self.n_vq + 1), self.audio_pad, dtype=np.int64)
        rows[:, 0] = text_ids
        if ref_codes is None:
            return rows
        rc = np.asarray(ref_codes, dtype=np.int64)
        ref = np.full((rc.shape[0], self.n_vq + 1), self.audio_pad, dtype=np.int64)
        ref[:, 0] = self.ref_slot
        ref[:, 1:] = rc
        return np.concatenate([rows, ref], axis=0)

    # ── acoustic frame: L_loc-layer cached ONNX steps + numpy heads/sampling ───
    def _empty_past(self):
        empty = np.zeros((1, self.nH_loc, 0, self.hd_loc), dtype=np.float32)
        feed = {}
        for i in range(self.L_loc):
            feed[f"past_k_{i}"] = empty
            feed[f"past_v_{i}"] = empty
        return feed

    @staticmethod
    def _split_past(out, L):
        # out = [hidden, present_k_0..L-1, present_v_0..L-1]
        pk = out[1:1 + L]
        pv = out[1 + L:1 + 2 * L]
        return pk, pv

    def _past_feed(self, pk, pv):
        feed = {}
        for i in range(self.L_loc):
            feed[f"past_k_{i}"] = pk[i]
            feed[f"past_v_{i}"] = pv[i]
        return feed

    def _acoustic_frame(self, h, temperature, top_k, top_p, rep_pen, hist):
        H = self.hidden
        cond = h[0].astype(np.float32)
        txt = self.text_emb[self.sgs].astype(np.float32)
        tok = np.stack([cond, txt])[None].astype(np.float32)        # (1,2,H)
        feed = {"token_emb": tok, "position_ids": np.array([[0, 1]], np.int64)}
        feed.update(self._empty_past())
        out = self.sess_ac.run(None, feed)
        hidden = out[0]
        pk, pv = self._split_past(out, self.L_loc)
        slot0 = hidden[0, 0]

        def samp(ch, vec):
            logits = vec.astype(np.float32) @ self.audio_emb[ch].T   # (Va,)
            prev = hist[ch] if hist is not None else None
            code = self._sample(logits, temperature, top_k, top_p, rep_pen, prev)
            if hist is not None:
                hist[ch].add(code)
            return code

        codes = [samp(0, hidden[0, 1])]
        for ch in range(1, self.n_vq):
            emb = self.audio_emb[ch - 1][codes[-1]].astype(np.float32)
            feed = {"token_emb": emb.reshape(1, 1, H),
                    "position_ids": np.array([[ch + 1]], np.int64)}
            feed.update(self._past_feed(pk, pv))
            out = self.sess_ac.run(None, feed)
            hidden = out[0]
            pk, pv = self._split_past(out, self.L_loc)
            codes.append(samp(ch, hidden[0, 0]))
        text_logits = slot0.astype(np.float32) @ self.text_emb.T
        eos = int(text_logits.argmax()) == self.eos_speech
        return codes, eos

    # ── Reference enrollment (cloning) ─────────────────────────────────────────
    def extract_speaker_emb(self, ref_audio, sr: Optional[int] = None) -> np.ndarray:
        wav, sr = self._load_mono(ref_audio, sr)
        return self._ensure_speaker_encoder().embed(wav, sr)

    def prepare_reference(self, ref_audio, *, sr: Optional[int] = None, denoise: bool = True,
                          use_ref_codes: bool = True, max_seconds: float = _MAX_REF_SECONDS
                          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Enroll a voice: trim → optional denoise → ``(speaker_emb, ref_codes)``."""
        wav, sr = self._load_mono(ref_audio, sr)                    # (n,) float32
        if wav.shape[-1] > int(max_seconds * sr):
            wav = wav[: int(max_seconds * sr)]
        if denoise and self.denoiser is not None:
            wav = self.denoiser.denoise(wav, sr)                    # -> float32 @ 44100
            sr = 44100
        speaker_emb = self._ensure_speaker_encoder().embed(wav, sr) if self.use_speaker_embedding else None
        ref_codes = self._encode_ref_wav(wav, sr) if use_ref_codes else None
        return speaker_emb, ref_codes

    # ── Public synthesis ───────────────────────────────────────────────────────
    def infer(self, phonemes: Optional[str] = None, text: str = "", ref_codes=None,
              speaker_emb=None, style: str = "tu_nhien", use_ref_codes: bool = True,
              ref_audio=None, ref_text=None, ref_phonemes=None,
              temperature: float = 0.8, top_k: int = 25, top_p: float = 0.95,
              max_new_frames: int = 300, repetition_penalty: float = 1.2, **_kw):
        if ref_codes is None and ref_audio is not None:
            speaker_emb, ref_codes = self.prepare_reference(ref_audio, use_ref_codes=use_ref_codes)
        if phonemes is None:
            from vieneu_utils.phonemize_text import phonemize_text_with_emotions
            phonemes = phonemize_text_with_emotions(text)
        if not use_ref_codes:
            ref_codes = None
        style_id = self._resolve_style_id(style)
        anchor = self._speaker_anchor(speaker_emb)
        rows = self._build_rows(phonemes, ref_codes, style_id)
        prompt_embeds = self._embed_rows(rows, anchor)              # (1, T, H)

        with self._lock:
            pre = self.sess_pre.run(None, {"inputs_embeds": prompt_embeds})
            past_k = [pre[1 + i] for i in range(self.L)]
            past_v = [pre[1 + self.L + i] for i in range(self.L)]
            h = pre[0][:, -1]
            Tprompt = prompt_embeds.shape[1]
            hist = [set() for _ in range(self.n_vq)] if not math.isclose(repetition_penalty, 1.0) else None
            frames: List[np.ndarray] = []
            for t in range(max_new_frames):
                codes, eos = self._acoustic_frame(h, temperature, top_k, top_p, repetition_penalty, hist)
                frames.append(np.asarray(codes, dtype=np.int64))
                if eos:
                    break
                slot = np.full((1, 1, self.n_vq + 1), self.audio_pad, dtype=np.int64)
                slot[:, :, 0] = self.sgs
                slot[0, 0, 1:] = codes
                se = self._embed_rows(slot[0], anchor)              # (1,1,H)
                feed = {"inputs_embeds": se, "position_ids": np.array([[Tprompt + t]], np.int64)}
                for i in range(self.L):
                    feed[f"past_k_{i}"] = past_k[i]
                    feed[f"past_v_{i}"] = past_v[i]
                out = self.sess_dec.run(None, feed)
                h = out[0][:, 0]
                past_k = [out[1 + i] for i in range(self.L)]
                past_v = [out[1 + self.L + i] for i in range(self.L)]

        if not frames:
            return np.zeros(0, dtype=np.float32)
        return self._decode_codes(np.stack(frames))                # (T, n_vq) → wav

    # ── streaming synthesis (native, frame-level) ──────────────────────────────
    def _stream_new_state(self) -> dict:
        """Fresh decode_step state (cached_positions init = -1 marks empty slots)."""
        sd = self._codec_stream_spec
        st: dict = {}
        for t in sd["transformer_offsets"]:
            st[t["input_name"]] = np.zeros(tuple(t["shape"]), np.int32)
        for a in sd["attention_caches"]:
            st[a["offset_input_name"]] = np.zeros(tuple(a["offset_shape"]), np.int32)
            st[a["cached_keys_input_name"]] = np.zeros(tuple(a["cache_shape"]), np.float32)
            st[a["cached_values_input_name"]] = np.zeros(tuple(a["cache_shape"]), np.float32)
            st[a["cached_positions_input_name"]] = np.full(tuple(a["positions_shape"]), -1, np.int32)
        return st

    def _stream_decode(self, frames: np.ndarray, state: dict) -> np.ndarray:
        """Decode a group of frames (K, n_vq) incrementally; threads `state` in place."""
        sd = self._codec_stream_spec
        codes = np.asarray(frames, dtype=np.int32)[None]           # (1, K, n_vq)
        feed = {"audio_codes": codes, "audio_code_lengths": np.array([codes.shape[1]], np.int32)}
        feed.update(state)
        outs = self.sess_codec_step.run(None, feed)
        d = dict(zip(self._codec_step_out_names, outs))
        for t in sd["transformer_offsets"]:
            state[t["input_name"]] = d[t["output_name"]]
        for a in sd["attention_caches"]:
            state[a["offset_input_name"]] = d[a["offset_output_name"]]
            state[a["cached_keys_input_name"]] = d[a["cached_keys_output_name"]]
            state[a["cached_values_input_name"]] = d[a["cached_values_output_name"]]
            state[a["cached_positions_input_name"]] = d[a["cached_positions_output_name"]]
        return d["audio"][0].mean(0)[: int(d["audio_lengths"][0])].astype(np.float32)

    def infer_stream(self, phonemes: Optional[str] = None, text: str = "", ref_codes=None,
                     speaker_emb=None, style: str = "tu_nhien", use_ref_codes: bool = True,
                     temperature: float = 0.8, top_k: int = 25, top_p: float = 0.95,
                     max_new_frames: int = 300, chunk_frames: int = 25,
                     repetition_penalty: float = 1.2, **_kw) -> Generator[np.ndarray, None, None]:
        """Native low-latency streaming: yields 48 kHz audio as frames are produced.

        Uses the MOSS streaming codec (decode_step), which is bit-exact to the full
        decoder, so streamed audio matches non-streaming quality with no boundary
        clicks. Falls back to a single ``infer`` if the streaming codec is missing.
        """
        if self.sess_codec_step is None or self._codec_stream_spec is None:
            yield self.infer(phonemes=phonemes, text=text, ref_codes=ref_codes,
                             speaker_emb=speaker_emb, style=style, use_ref_codes=use_ref_codes,
                             temperature=temperature, top_k=top_k, top_p=top_p,
                             max_new_frames=max_new_frames, repetition_penalty=repetition_penalty)
            return
        if phonemes is None:
            from vieneu_utils.phonemize_text import phonemize_text_with_emotions
            phonemes = phonemize_text_with_emotions(text)
        if not use_ref_codes:
            ref_codes = None
        style_id = self._resolve_style_id(style)
        anchor = self._speaker_anchor(speaker_emb)
        rows = self._build_rows(phonemes, ref_codes, style_id)
        prompt_embeds = self._embed_rows(rows, anchor)

        # Acquire initial prefill under lock, then do per-iteration ONNX calls
        with self._lock:
            pre = self.sess_pre.run(None, {"inputs_embeds": prompt_embeds})
            past_k = [pre[1 + i] for i in range(self.L)]
            past_v = [pre[1 + self.L + i] for i in range(self.L)]
            h = pre[0][:, -1]
            Tprompt = prompt_embeds.shape[1]
            hist = [set() for _ in range(self.n_vq)] if not math.isclose(repetition_penalty, 1.0) else None

        state = self._stream_new_state()
        buffer: List[np.ndarray] = []
        sr = self.SAMPLE_RATE
        emitted = 0
        t_first: Optional[float] = None

        def _target() -> int:
            cap = max(1, chunk_frames)
            if t_first is None:
                return min(cap, _STREAM_LEADIN_FRAMES)
            lead = emitted / sr - (time.perf_counter() - t_first)
            if lead < 0.20:
                return min(cap, _STREAM_LEADIN_FRAMES)
            if lead < 0.55:
                return min(cap, 6)
            if lead < 1.10:
                return min(cap, 8)
            return cap

        for t in range(max_new_frames):
            wav_chunk = None
            # protect ONNX session calls per-iteration; release before yielding
            with self._lock:
                codes, eos = self._acoustic_frame(h, temperature, top_k, top_p, repetition_penalty, hist)
                buffer.append(np.asarray(codes, dtype=np.int64))
                if not eos:
                    slot = np.full((1, 1, self.n_vq + 1), self.audio_pad, dtype=np.int64)
                    slot[:, :, 0] = self.sgs
                    slot[0, 0, 1:] = codes
                    se = self._embed_rows(slot[0], anchor)
                    feed = {"inputs_embeds": se, "position_ids": np.array([[Tprompt + t]], np.int64)}
                    for i in range(self.L):
                        feed[f"past_k_{i}"] = past_k[i]
                        feed[f"past_v_{i}"] = past_v[i]
                    out = self.sess_dec.run(None, feed)
                    h = out[0][:, 0]
                    past_k = [out[1 + i] for i in range(self.L)]
                    past_v = [out[1 + self.L + i] for i in range(self.L)]
                if len(buffer) >= _target() or eos:
                    wav_chunk = self._stream_decode(np.stack(buffer), state)
                    buffer = []

            # yield after the lock is released
            if wav_chunk is not None and wav_chunk.size:
                if t_first is None:
                    t_first = time.perf_counter()
                emitted += wav_chunk.size
                yield wav_chunk
            if eos:
                break

        if buffer:
            with self._lock:
                wav = self._stream_decode(np.stack(buffer), state)
            if wav.size:
                yield wav

    # ── codec (MOSS ONNX) ──────────────────────────────────────────────────────
    def _decode_codes(self, codes: np.ndarray) -> np.ndarray:
        c = np.asarray(codes, dtype=np.int32)[None]                # (1, T, n_vq) — codec wants int32
        lens = np.array([c.shape[1]], dtype=np.int32)
        out = self.sess_codec_dec.run(None, {"audio_codes": c, "audio_code_lengths": lens})
        return out[0][0].mean(0).astype(np.float32)

    def _encode_ref_wav(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """wav: 1D mono float → MOSS ref codes (T, n_vq), torch-free."""
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if sr != self.SAMPLE_RATE:
            import soxr
            wav = soxr.resample(wav, sr, self.SAMPLE_RATE).astype(np.float32)
        stereo = np.stack([wav, wav])[None].astype(np.float32)     # (1, 2, n)
        lens = np.array([stereo.shape[-1]], dtype=np.int32)
        if self._sess_codec_enc is None:
            import onnxruntime as ort
            self._sess_codec_enc = ort.InferenceSession(
                self._codec_enc_path,
                providers=self.onnx_providers,
            )
        out = self._sess_codec_enc.run(None, {"waveform": stereo, "input_lengths": lens})
        return np.asarray(out[0][0], dtype=np.int64)               # (T, n_vq)

    def _encode_ref(self, ref_audio_path: str) -> np.ndarray:
        wav, sr = self._load_mono(ref_audio_path, None)
        return self._encode_ref_wav(wav, sr)

    # ── audio I/O (soundfile, torch-free) ──────────────────────────────────────
    def _load_mono(self, ref_audio: Union[str, np.ndarray], sr: Optional[int]) -> Tuple[np.ndarray, int]:
        if isinstance(ref_audio, (str, bytes)) or hasattr(ref_audio, "__fspath__"):
            import soundfile as sf
            wav, sr = sf.read(str(ref_audio), dtype="float32", always_2d=True)  # (n, ch)
            wav = wav.mean(axis=1)                                  # → mono (n,)
        else:
            wav = np.asarray(ref_audio, dtype=np.float32)
            if sr is None:
                raise ValueError("Pass `sr` when giving a waveform array.")
            if wav.ndim == 2:
                wav = wav.mean(axis=0) if wav.shape[0] <= wav.shape[1] else wav.mean(axis=1)
        return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)
