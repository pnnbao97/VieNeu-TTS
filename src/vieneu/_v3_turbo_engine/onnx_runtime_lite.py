"""
VieNeu-TTS v3 Turbo — torch-free ONNX engine (CPU).
===================================================
A fully PyTorch-free inference engine: the transformer forwards run in ONNX
Runtime, the MOSS audio codec runs in ONNX Runtime, and everything else
(embeddings, output heads, sampling, prompt build) is plain NumPy. The only deps
are onnxruntime, numpy, tokenizers, sea-g2p, soundfile, huggingface_hub.

Artifacts (all fetched from HF, or a local dir):
  v3 graphs  : <repo>/onnx/{vieneu_prefill,vieneu_decode_step,vieneu_acoustic_cached}.onnx
               + vieneu_backbone_shared.data  + vieneu_v3_heads.npz (tied embeddings/heads)
  text setup : <repo>/{config.json, tokenizer.json}
  codec      : <codec_repo>/{moss_audio_tokenizer_decode_full,_encode}.onnx (+ .data)

API matches VieNeuTTSv3Turbo.infer(...) so V3TurboVieNeuTTS can use it as a
drop-in engine on CPU.
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

_V3_REPO = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
_V3_FILES = [
    "onnx/vieneu_prefill.onnx", "onnx/vieneu_decode_step.onnx",
    "onnx/vieneu_acoustic_cached.onnx", "onnx/vieneu_backbone_shared.data",
    "onnx/vieneu_v3_heads.npz",
]
_CODEC_FILES = [
    "moss_audio_tokenizer_decode_full.onnx", "moss_audio_tokenizer_decode_shared.data",
    "moss_audio_tokenizer_encode.onnx", "moss_audio_tokenizer_encode.data",
]


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
        codec_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        threads: int = 0,
        **_kw,
    ):
        import onnxruntime as ort

        self._lock = threading.RLock()
        self.device = _Dev()
        repo = onnx_repo or checkpoint_path

        # ── Fetch artifacts ────────────────────────────────────────────────
        if onnx_dir:
            vd = Path(onnx_dir)
            npz_path = vd / "vieneu_v3_heads.npz"
        else:
            vd = self._fetch(repo, _V3_FILES, hf_token)          # cached .../onnx/
            npz_path = vd / "vieneu_v3_heads.npz"
        cd = Path(codec_dir) if codec_dir else self._fetch(codec_repo, _CODEC_FILES, hf_token)

        cfg_path = self._fetch_file(repo, "config.json", hf_token, onnx_dir, "config.json")
        tok_path = self._fetch_file(repo, "tokenizer.json", hf_token, onnx_dir, "tokenizer.json")

        # ── Config (token ids etc.) ────────────────────────────────────────
        c = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        self.n_vq = int(c["n_vq"])
        self.hidden = int(c["hidden_size"])
        self.L = int(c["num_hidden_layers"])
        self.audio_pad = int(c["audio_pad_token_id"])
        self.tps = int(c["text_prompt_start_token_id"])
        self.tpe = int(c["text_prompt_end_token_id"])
        self.sgs = int(c["speech_generation_start_token_id"])
        self.eos_speech = int(c["speech_generation_end_token_id"])
        self.ref_slot = int(c["audio_ref_slot_token_id"])
        self.emotion_0 = int(c["emotion_0_token_id"])
        self.emotion_4 = int(c["emotion_4_token_id"])
        self.text_vocab = int(c["text_vocab_size"])

        # ── Tied embeddings/heads (numpy) ──────────────────────────────────
        z = np.load(npz_path)
        self.text_emb = z["text_emb"].astype(np.float32)            # (Vt, H)
        self.audio_emb = z["audio_emb"].astype(np.float32)          # (n_vq, Va, H)

        # ── Tokenizer (tokenizers lib, torch-free) ─────────────────────────
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tok_path))

        # ── ONNX sessions ──────────────────────────────────────────────────
        so = ort.SessionOptions()
        if threads and threads > 0:
            so.intra_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        self.sess_pre = ort.InferenceSession(str(vd / "vieneu_prefill.onnx"), so, providers=prov)
        self.sess_dec = ort.InferenceSession(str(vd / "vieneu_decode_step.onnx"), so, providers=prov)
        self.sess_ac = ort.InferenceSession(str(vd / "vieneu_acoustic_cached.onnx"), so, providers=prov)
        self.sess_codec_dec = ort.InferenceSession(
            str(cd / "moss_audio_tokenizer_decode_full.onnx"), so, providers=prov)
        self._codec_enc_path = str(cd / "moss_audio_tokenizer_encode.onnx")
        self._sess_codec_enc = None  # lazy (only for cloning)
        self.nH_loc = int(c.get("local_num_attention_heads", 8))
        self.hd_loc = self.hidden // self.nH_loc

    # ── artifact helpers ──────────────────────────────────────────────────
    @staticmethod
    def _fetch(repo: str, files: List[str], hf_token: Optional[str]) -> Path:
        from huggingface_hub import hf_hub_download
        last = None
        for fn in files:
            last = hf_hub_download(repo, fn, repo_type="model", token=hf_token)
        return Path(last).parent

    @staticmethod
    def _fetch_file(repo, fn, hf_token, onnx_dir, local_name):
        # Prefer a local copy next to onnx_dir's parent if present, else HF.
        if onnx_dir:
            cand = Path(onnx_dir) / local_name
            if cand.exists():
                return cand
            cand = Path(onnx_dir).parent / local_name
            if cand.exists():
                return cand
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo, fn, repo_type="model", token=hf_token)

    # ── numpy embedding / heads / sampling ────────────────────────────────
    def _embed_rows(self, rows: np.ndarray) -> np.ndarray:
        """rows: (T, n_vq+1) int → (1, T, H) float32 (mirror _build_inputs_embeds)."""
        emb = self.text_emb[rows[:, 0]]                              # (T, H)
        for ch in range(self.n_vq):
            ids = rows[:, ch + 1]
            valid = ids != self.audio_pad
            safe = np.where(valid, ids, 0)
            emb = emb + self.audio_emb[ch][safe] * valid[:, None]
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
        if top_k and top_k > 0:
            k = min(int(top_k), logits.shape[-1])
            kth = np.partition(logits, -k)[-k]
            logits = np.where(logits < kth, -np.inf, logits)
        if top_p and top_p < 1.0:
            order = np.argsort(logits)[::-1]
            s = logits[order]
            p = _softmax(s)
            remove = (np.cumsum(p) - p) > top_p
            s = np.where(remove, -np.inf, s)
            out = np.full_like(logits, -np.inf)
            out[order] = s
            logits = out
        p = _softmax(logits)
        return int(np.random.choice(p.shape[-1], p=p))

    # ── prompt build (numpy, mirror prompt_v3_turbo.build_prompt_2d) ───────
    def _build_rows(self, phonemes: str, ref_codes: Optional[np.ndarray], emo_token: int) -> np.ndarray:
        phone_ids = self.tokenizer.encode(phonemes, add_special_tokens=False).ids
        text_ids = [emo_token, self.tps] + list(phone_ids) + [self.tpe]
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

    def _leading_token(self, emotion, voice_token_id):
        if voice_token_id is not None:
            return int(voice_token_id)
        return self.emotion_0 if emotion == "natural" else self.emotion_4

    # ── acoustic frame: 16 cached ONNX steps + numpy heads/sampling ───────
    def _acoustic_frame(self, h, temperature, top_k, top_p, rep_pen, hist):
        H, nH, hd = self.hidden, self.nH_loc, self.hd_loc
        empty = np.zeros((1, nH, 0, hd), dtype=np.float32)
        cond = h[0].astype(np.float32)
        txt = self.text_emb[self.sgs].astype(np.float32)
        tok = np.stack([cond, txt])[None].astype(np.float32)        # (1,2,H)
        o = self.sess_ac.run(None, {
            "token_emb": tok, "position_ids": np.array([[0, 1]], np.int64),
            "past_k_0": empty, "past_k_1": empty, "past_v_0": empty, "past_v_1": empty})
        hidden, pk0, pk1, pv0, pv1 = o
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
            o = self.sess_ac.run(None, {
                "token_emb": emb.reshape(1, 1, H), "position_ids": np.array([[ch + 1]], np.int64),
                "past_k_0": pk0, "past_k_1": pk1, "past_v_0": pv0, "past_v_1": pv1})
            hidden, pk0, pk1, pv0, pv1 = o
            codes.append(samp(ch, hidden[0, 0]))
        text_logits = slot0.astype(np.float32) @ self.text_emb.T
        eos = int(text_logits.argmax()) == self.eos_speech
        return codes, eos

    # ── Public API ────────────────────────────────────────────────────────
    def infer(self, text: str = "", ref_audio=None, ref_codes=None, ref_text=None,
              phonemes: Optional[str] = None, ref_phonemes=None, emotion: str = "natural",
              voice_token_id: Optional[int] = None, temperature: float = 0.8, top_k: int = 25,
              top_p: float = 0.95, max_new_frames: int = 300, repetition_penalty: float = 1.2):
        if ref_codes is None and ref_audio is not None:
            ref_codes = self._encode_ref(ref_audio)
        if phonemes is None:
            from vieneu_utils.phonemize_text import phonemize_text_with_emotions
            phonemes = phonemize_text_with_emotions(text)
        emo = self._leading_token(emotion, voice_token_id)
        rows = self._build_rows(phonemes, ref_codes, emo)
        prompt_embeds = self._embed_rows(rows)                       # (1, T, H)

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
                se = self._embed_rows(slot[0])                       # (1,1,H)
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
        return self._decode_codes(np.stack(frames))                 # (T, n_vq) → wav

    # ── codec (MOSS ONNX) ─────────────────────────────────────────────────
    def _decode_codes(self, codes: np.ndarray) -> np.ndarray:
        """codes (T, n_vq) int → float32 mono waveform."""
        c = np.asarray(codes, dtype=np.int32)[None]                 # (1, T, n_vq) — codec wants int32
        lens = np.array([c.shape[1]], dtype=np.int32)
        out = self.sess_codec_dec.run(None, {"audio_codes": c, "audio_code_lengths": lens})
        audio = out[0]                                              # (1, ch, samples)
        return audio[0].mean(0).astype(np.float32)

    def _encode_ref(self, ref_audio_path: str) -> np.ndarray:
        """Encode a wav → MOSS ref codes (T, n_vq), torch-free."""
        import soundfile as sf
        wav, sr = sf.read(str(ref_audio_path), dtype="float32", always_2d=True)  # (n, ch)
        wav = wav.T                                                 # (ch, n)
        if sr != self.SAMPLE_RATE:
            import soxr
            wav = np.stack([soxr.resample(wav[c], sr, self.SAMPLE_RATE) for c in range(wav.shape[0])])
        if wav.shape[0] == 1:
            wav = np.repeat(wav, 2, axis=0)                         # mono → stereo
        else:
            wav = wav[:2]
        wav = wav[None].astype(np.float32)                         # (1, 2, n)
        lens = np.array([wav.shape[-1]], dtype=np.int32)
        if self._sess_codec_enc is None:
            import onnxruntime as ort
            self._sess_codec_enc = ort.InferenceSession(self._codec_enc_path, providers=["CPUExecutionProvider"])
        out = self._sess_codec_enc.run(None, {"waveform": wav, "input_lengths": lens})
        return np.asarray(out[0][0], dtype=np.int64)               # (T, n_vq)
