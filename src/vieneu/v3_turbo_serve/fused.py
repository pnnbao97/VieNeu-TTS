"""
One CUDA graph per audio frame: acoustic decoder + sampling + repetition
penalty + EOS check + backbone decode step, for a whole batch.

Why
---
Measured on an RTX 3060 (batch of 6 chunks), a frame of the plain loop cost
~65 ms while the GPU sat mostly idle: ~45 ms in the acoustic step — of which
22 ms was the repetition penalty being applied on the HOST (one ``.item()``
sync and one host→device copy per row per codebook, 16 × B per frame) — and
~19 ms in the HF backbone step (Python/HF overhead and a ``torch.cat`` of the
attention mask every step; the maths itself is ~2 ms). The batch size hardly
mattered: 2 chunks took 8.4 s, 16 chunks 15 s. The loop was latency-bound.

The existing ``CudaGraphedFrame`` captured only the acoustic step and was
disabled whenever ``repetition_penalty != 1.0`` — i.e. always, in practice.

What
----
* ``GpuRepHistory`` — the sliding-window repetition history as tensors
  (per-row, per-codebook counts + a ring buffer of the last ``window``
  codes). Same rule as ``RepetitionHistory``: a code seen in the last
  ``window`` frames is penalised (``logit<0 → *p, else /p``) before
  temperature. No host round-trips, so it can live inside a graph.
* ``StaticBackbone`` — the Qwen3 decode step over a preallocated KV cache
  and a fixed-shape boolean mask, built from the HF modules' own weights
  (norms, projections, rotary and MLP are the HF modules themselves; only the
  cache plumbing and the attention call are ours). No ``DynamicCache``, no
  growing mask, so the step is graph-capturable.
* ``FusedFrame`` — captures ``acoustic frame → codes/EOS → bookkeeping →
  backbone step`` once per (batch size, sampling settings, cache length) and
  replays it once per frame. Codes, per-row lengths and the finished flags
  stay on the device; the host reads one boolean per frame ("all done?").

``torch.compile`` is deliberately not used: Inductor needs a C++ toolchain,
which the app's Windows clients do not have. ``torch.cuda.CUDAGraph`` needs
nothing.

Exactness: the backbone step matches the HF path to bf16 noise (checked
against ``BatchedBackbone.decode_step`` on the same cache); sampling is the
same distribution as ``_sample_batched`` but draws differently, as the batch
path already differs from the single path.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .._v3_turbo_engine.capture_lock import CAPTURE_LOCK


# ── repetition history on the device ─────────────────────────────────────────

class GpuRepHistory:
    """Sliding-window repetition history for ``B`` rows × ``n_vq`` codebooks.

    ``counts[b, ch, code]`` is how many of the last ``window`` frames of row
    ``b`` produced ``code`` on codebook ``ch``; ``ring[b, ch, slot]`` remembers
    which code each frame produced so it can be un-counted when it leaves the
    window. ``frame`` is a device scalar so the whole update is graph-safe.
    ``window <= 0`` keeps everything (the pre-window behaviour).
    """

    def __init__(self, B: int, n_vq: int, vocab: int, window: int, device):
        self.window = int(window)
        self.counts = torch.zeros(B, n_vq, vocab, dtype=torch.int32, device=device)
        self.ring = torch.zeros(B, n_vq, max(self.window, 1), dtype=torch.long, device=device)
        self.frame = torch.zeros((), dtype=torch.long, device=device)
        # Frame at which each row's history began: a row that joins a running
        # batch (stream.py) must not evict ring slots it never wrote.
        self.start = torch.zeros(B, dtype=torch.long, device=device)
        self._ones = torch.ones(B, 1, dtype=torch.int32, device=device)

    def reset(self) -> None:
        self.counts.zero_()
        self.ring.zero_()
        self.frame.zero_()
        self.start.zero_()

    def reset_rows(self, rows: torch.Tensor) -> None:
        """Clear the history of ``rows`` (long index tensor); they start now."""
        self.counts[rows] = 0
        self.ring[rows] = 0
        self.start[rows] = self.frame

    def penalise(self, logits: torch.Tensor, ch: int, penalty) -> torch.Tensor:
        """``logits`` (B, V) float → penalised copy. ``penalty`` is a float or
        a per-row ``(B, 1)`` tensor."""
        seen = self.counts[:, ch] > 0
        bent = torch.where(logits < 0, logits * penalty, logits / penalty)
        return torch.where(seen, bent, logits)

    def add(self, ch: int, code: torch.Tensor) -> None:
        """Record this frame's ``code`` (B,) for codebook ``ch``."""
        counts = self.counts[:, ch]
        if self.window > 0:
            slot = torch.remainder(self.frame, self.window).view(1)
            # The code written to this slot ``window`` frames ago leaves the
            # window now — but only once the ring has actually wrapped.
            old = self.ring[:, ch].index_select(1, slot)              # (B, 1)
            evict = ((self.frame - self.start) >= self.window).to(torch.int32).view(-1, 1)
            counts.scatter_add_(1, old, -(self._ones * evict))
            self.ring[:, ch].index_copy_(1, slot, code.view(-1, 1))
        counts.scatter_add_(1, code.view(-1, 1), self._ones)

    def advance(self) -> None:
        self.frame += 1


def sample_gpu(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Top-k + top-p sampling over a batch. ``logits`` (B, V) → codes (B,).

    The same steps as ``batched_acoustic._sample_batched`` minus the host-side
    penalty (done by ``GpuRepHistory.penalise`` before this).
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-6)
    if top_k and 0 < top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if 0.0 < top_p < 1.0:
        s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(s_logits, dim=-1)
        drop = probs.cumsum(dim=-1) > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        s_logits = s_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, s_idx, s_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_gpu_rows(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor,
                    top_p: torch.Tensor) -> torch.Tensor:
    """``sample_gpu`` with per-row settings: ``temperature``/``top_p`` are
    ``(B, 1)`` floats, ``top_k`` is ``(B, 1)`` long (``<= 0`` disables). A row
    with ``temperature <= 0`` is greedy. Rows of a continuous batch carry their
    own request's sampling, so nothing here is baked into the graph."""
    V = logits.shape[-1]
    # Dead rows of a continuous batch keep stepping on stale state; keep the
    # multinomial below well-defined whatever they produce.
    logits = torch.nan_to_num(logits)
    greedy = logits.argmax(dim=-1)
    scaled = logits / temperature.clamp(min=1e-6)
    s_logits, s_idx = torch.sort(scaled, descending=True, dim=-1)
    kth = s_logits.gather(1, (top_k - 1).clamp(0, V - 1))
    kth = torch.where(top_k > 0, kth, torch.full_like(kth, float("-inf")))
    probs = F.softmax(s_logits, dim=-1)
    drop = probs.cumsum(dim=-1) > top_p
    drop[..., 1:] = drop[..., :-1].clone()
    drop[..., 0] = False
    drop |= s_logits < kth
    s_logits = s_logits.masked_fill(drop, float("-inf"))
    kept = torch.full_like(scaled, float("-inf")).scatter_(-1, s_idx, s_logits)
    sampled = torch.multinomial(F.softmax(kept, dim=-1), num_samples=1).squeeze(-1)
    return torch.where(temperature.view(-1) <= 0, greedy, sampled)


@torch.no_grad()
def acoustic_frame_gpu(
    model,
    backbone_hidden: torch.Tensor,        # (B, H)
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    rep: Optional[GpuRepHistory],
    sample_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One acoustic frame for B rows: ``(codes (B, n_vq), is_eos (B,))``.

    ``generate_frame_batched`` with the penalty on the device. Everything in
    here is a fixed sequence of device ops, so it captures into a CUDA graph.
    ``sample_fn(logits, ch) -> codes`` replaces penalty + sampling when given
    (the continuous batch samples each row with its own settings).
    """
    cfg = model.config
    n_vq, H = cfg.n_vq, cfg.hidden_size
    dec = model.acoustic_decoder
    L = len(dec.layers)
    dt = next(dec.parameters()).dtype
    dev = backbone_hidden.device
    B = backbone_hidden.shape[0]
    use_rep = rep is not None and not math.isclose(repetition_penalty, 1.0)

    def _sample_ch(ch: int, vec: torch.Tensor) -> torch.Tensor:
        logits = model.audio_lm_heads[ch](vec).float()                            # (B, V)
        if sample_fn is not None:
            return sample_fn(logits, ch)
        if use_rep:
            logits = rep.penalise(logits, ch, repetition_penalty)
        code = sample_gpu(logits, temperature, top_k, top_p)
        if use_rep:
            rep.add(ch, code)
        return code

    cond = backbone_hidden.to(dt)
    sgs_ids = torch.full((B,), cfg.speech_generation_start_token_id, device=dev, dtype=torch.long)
    txt = model.text_embeddings(sgs_ids).to(dt)
    tok = torch.stack([cond, txt], dim=1)                                          # (B, 2, H)
    pos = torch.arange(2, device=dev, dtype=torch.long)
    hidden, pk, pv = dec.cached_step(tok, pos, [None] * L, [None] * L)
    is_eos = model.text_lm_head(hidden[:, 0]).float().argmax(-1) == cfg.speech_generation_end_token_id

    codes: List[torch.Tensor] = [_sample_ch(0, hidden[:, 1])]
    for ch in range(1, n_vq):
        emb = model.audio_embeddings[ch - 1](codes[-1]).to(dt)                     # (B, H)
        pos = torch.arange(ch + 1, ch + 2, device=dev, dtype=torch.long)
        hidden, pk, pv = dec.cached_step(emb.view(B, 1, H), pos, pk, pv)
        codes.append(_sample_ch(ch, hidden[:, 0]))
    if use_rep:
        rep.advance()
    return torch.stack(codes, dim=1), is_eos


# ── backbone decode over a static cache ──────────────────────────────────────

class StaticBackbone:
    """Qwen3 decode step for a batch over a preallocated KV cache.

    Prompts are left-padded (see ``BatchedBackbone.prefill``), so every row
    writes its new token at the same cache index ``cur`` and only the mask
    differs per row. The HF modules do the arithmetic; this class only owns
    the cache, the mask and the positions, all of which update on the device.

    The cache is a ring: index ``cur mod max_len``. A batch loaded with
    ``load`` never wraps (its bucket covers prompt + frames); a continuous
    batch (``load_row``, stream.py) runs indefinitely and relies on a row's
    prompt + frames being shorter than ``max_len``, so the write index never
    re-enters a live row's range. Rotary positions are per row and
    independent of the cache index.
    """

    def __init__(self, model, B: int, max_len: int):
        bb = model.semantic_backbone
        cfg = bb.config
        p = next(bb.parameters())
        self.layers = bb.layers
        self.norm = bb.norm
        self.rotary = bb.rotary_emb
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.B, self.max_len = B, int(max_len)
        L = len(self.layers)
        dev, dt = p.device, p.dtype
        self.k = torch.zeros(L, B, self.n_kv, self.max_len, self.hd, device=dev, dtype=dt)
        self.v = torch.zeros_like(self.k)
        # Additive attention bias, (B, 1, 1, max_len): 0 where a row may attend,
        # -inf elsewhere. A float bias (not a bool mask) keeps SDPA on the
        # memory-efficient kernel; measured 1.29 ms → 0.10 ms per layer.
        self.bias = torch.full((B, 1, 1, self.max_len), float("-inf"), device=dev, dtype=dt)
        # Next write index (shared: rows are right-aligned) and per-row position ids.
        self.cur = torch.zeros((), device=dev, dtype=torch.long)
        self.pos = torch.zeros(B, 1, device=dev, dtype=torch.long)

    @torch.no_grad()
    def load(self, cache, attn_mask: torch.Tensor, cur_pos: torch.Tensor) -> None:
        """Take over from an HF prefill: ``cache`` (DynamicCache), ``attn_mask``
        (B, T) and ``cur_pos`` (B,) as ``BatchedBackbone.prefill`` returns them."""
        T = attn_mask.shape[1]
        if T + 1 > self.max_len:
            raise ValueError(f"prompt of {T} tokens does not fit a cache of {self.max_len}")
        self.k.zero_()
        self.v.zero_()
        for i in range(len(self.layers)):
            layer = cache.layers[i] if hasattr(cache, "layers") else None
            k = layer.keys if layer is not None else cache.key_cache[i]
            v = layer.values if layer is not None else cache.value_cache[i]
            self.k[i, :, :, :T].copy_(k)
            self.v[i, :, :, :T].copy_(v)
        self.bias.fill_(float("-inf"))
        self.bias[:, 0, 0, :T].masked_fill_(attn_mask.bool(), 0.0)
        self.cur.fill_(T)
        self.pos.copy_((cur_pos + 1).view(-1, 1))

    @torch.no_grad()
    def load_row(self, b: int, keys: List[torch.Tensor], values: List[torch.Tensor]) -> None:
        """Put one prefilled prompt into slot ``b`` of a running batch.

        ``keys[i]``/``values[i]`` are layer ``i``'s ``(n_kv, T, hd)`` for the
        prompt's real tokens only. They land right-aligned to the write index,
        so the row's first frame goes to ``cur`` like everyone else's; its
        rotary positions restart at ``T``. Host-side (outside the graph).
        """
        T = keys[0].shape[1]
        if T + 1 > self.max_len:
            raise ValueError(f"prompt of {T} tokens does not fit a cache of {self.max_len}")
        idx = torch.remainder(self.cur - T + torch.arange(T, device=self.cur.device), self.max_len)
        for i in range(len(self.layers)):
            self.k[i, b].index_copy_(1, idx, keys[i].to(self.k.dtype))
            self.v[i, b].index_copy_(1, idx, values[i].to(self.v.dtype))
        self.bias[b].fill_(float("-inf"))
        self.bias[b, 0, 0].index_fill_(0, idx, 0.0)
        self.pos[b] = T

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (B, 1, H) for the new token → last hidden (B, H). Advances the cache."""
        B, g = self.B, self.n_heads // self.n_kv
        idx = torch.remainder(self.cur, self.max_len).view(1)
        # The new token attends to itself.
        self.bias.view(B, self.max_len).index_fill_(1, idx, 0.0)
        cos, sin = self.rotary(x, self.pos)
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_norm(a.q_proj(h).view(B, 1, self.n_heads, self.hd)).transpose(1, 2)
            k = a.k_norm(a.k_proj(h).view(B, 1, self.n_kv, self.hd)).transpose(1, 2)
            v = a.v_proj(h).view(B, 1, self.n_kv, self.hd).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            self.k[i][:, :, idx] = k
            self.v[i][:, :, idx] = v
            # GQA without repeating K/V: the g query heads that share a KV head
            # are folded into the query-length axis (each query row attends
            # independently, so this is exact) — plain MHA with q_len = g,
            # which the efficient kernel takes; ``enable_gqa`` fell back to
            # the math path and cost 12× more.
            attn = F.scaled_dot_product_attention(
                q.view(B, self.n_kv, g, self.hd), self.k[i], self.v[i],
                attn_mask=self.bias, scale=a.scaling,
            )
            x = x + a.o_proj(attn.reshape(B, 1, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = self.norm(x)
        self.cur += 1
        self.pos += 1
        return x[:, 0]


# ── the graph ────────────────────────────────────────────────────────────────

def cache_len_bucket(needed: int, max_positions: int, step: int = 512) -> int:
    """Round a cache length up so a handful of graphs cover every prompt."""
    return min(max_positions, int(math.ceil(needed / step)) * step)


def batch_bucket(n: int) -> int:
    """Pad a batch to a power of two so at most ~7 graphs exist per setting."""
    return 1 << max(0, (n - 1).bit_length())


class FusedFrame:
    """One captured graph: acoustic frame + sampling + bookkeeping + backbone step.

    Static state (all on the device):
      ``h`` (B, H)            backbone hidden feeding the acoustic decoder
      ``codes`` (F, B, n_vq)  every frame's codes, written at ``frame``
      ``finished`` (B,)       row has seen EOS or hit its cap
      ``length`` (B,)         frames kept for the row (EOS frame included)
      ``caps`` (B,)           per-row frame cap
      ``spk`` (B, D)          speaker anchor re-added at every step (or None)

    ``max_frames`` sizes the codes buffer; the per-call ``max_new_frames`` is
    just how many replays the host loop allows, so one graph serves every
    frame budget up to the buffer.
    """

    def __init__(self, model, B: int, max_len: int, max_frames: int, *,
                 temperature: float, top_k: int, top_p: float,
                 repetition_penalty: float, repetition_window: int, warmup: int = 3):
        self.model = model
        cfg = model.config
        p = next(model.parameters())
        dev, dt = p.device, p.dtype
        self.B, self.max_frames = B, int(max_frames)
        n_vq, H = cfg.n_vq, cfg.hidden_size
        self.sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p,
                             repetition_penalty=repetition_penalty)
        self.rep = (GpuRepHistory(B, n_vq, cfg.audio_vocab_size, repetition_window, dev)
                    if not math.isclose(repetition_penalty, 1.0) else None)
        self.bb = StaticBackbone(model, B, max_len)
        self.h = torch.zeros(B, H, device=dev, dtype=dt)
        self.codes = torch.zeros(self.max_frames, B, n_vq, device=dev, dtype=torch.long)
        self.finished = torch.zeros(B, device=dev, dtype=torch.bool)
        self.length = torch.zeros(B, device=dev, dtype=torch.long)
        self.caps = torch.full((B,), self.max_frames, device=dev, dtype=torch.long)
        self.frame = torch.zeros((), device=dev, dtype=torch.long)
        spk_dim = getattr(cfg, "speaker_embedding_dim", None)
        self.spk = (torch.zeros(B, spk_dim, device=dev, dtype=torch.float32)
                    if model.xvec_proj is not None and spk_dim else None)
        self._pad = cfg.audio_pad_token_id
        self._sgs = cfg.speech_generation_start_token_id
        self.all_done = torch.zeros((), device=dev, dtype=torch.bool)

        # Capture on a side stream after warm-up runs, as CUDA graphs require.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(warmup):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self._body()

    def _body(self) -> None:
        codes, is_eos = acoustic_frame_gpu(self.model, self.h, rep=self.rep, **self.sampling)
        # Bookkeeping, all on the device: a row's codes are kept up to and
        # including the frame that finished it.
        self.codes.index_copy_(0, self.frame.view(1), codes.unsqueeze(0))
        n = self.frame + 1
        done_now = ~self.finished & (is_eos | (n >= self.caps))
        self.length.copy_(torch.where(done_now, n, self.length))
        self.finished.logical_or_(done_now)
        self.all_done.copy_(self.finished.all())
        self.frame += 1
        # Feed the frame back as the next backbone input (also for finished
        # rows — their output is never read, and a static batch cannot shrink).
        B, n_vq = codes.shape
        slot = torch.full((B, 1, n_vq + 1), self._pad, dtype=torch.long, device=codes.device)
        slot[:, :, 0] = self._sgs
        slot[:, 0, 1:] = codes
        se = self.model._build_inputs_embeds(slot, speaker_emb=self.spk)
        self.h.copy_(self.bb.step(se))

    @torch.no_grad()
    def run(self, h0: torch.Tensor, cache, attn_mask, cur_pos, caps: List[int],
            spk: Optional[torch.Tensor], max_new_frames: int,
            on_frame: Optional[Callable[[], None]] = None) -> List[torch.Tensor]:
        """Generate for the rows loaded into the static buffers.

        ``h0`` (B, H) is the prefill's last hidden; ``cache``/``attn_mask``/
        ``cur_pos`` come from ``BatchedBackbone.prefill``; ``caps[b]`` is row
        ``b``'s own frame cap (already ≤ ``max_new_frames``). ``on_frame`` is
        called before every frame — a server's cancel check, which used to
        hang off the backbone step this loop no longer calls. Returns per-row
        codes ``(T_b, n_vq)`` on the device.

        The device work holds ``CAPTURE_LOCK`` shared, one frame at a time, so a
        capture in another thread (another graph) waits for at most a frame.
        Only the thread that owns this graph may call ``run``.
        """
        B = self.B
        with CAPTURE_LOCK.shared():
            self.bb.load(cache, attn_mask, cur_pos)
            self.h.copy_(h0)
            self.finished.zero_()
            self.length.zero_()
            self.frame.zero_()
            self.all_done.zero_()
            self.caps.copy_(torch.as_tensor(caps, dtype=torch.long, device=self.caps.device))
            if self.rep is not None:
                self.rep.reset()
            if self.spk is not None:
                if spk is None:
                    raise ValueError("this model needs a speaker anchor for every row")
                self.spk.copy_(spk.to(self.spk.dtype))
        for _ in range(min(int(max_new_frames), self.max_frames)):
            if on_frame is not None:
                on_frame()
            with CAPTURE_LOCK.shared():
                self.graph.replay()
                done = bool(self.all_done)
            if done:
                break
        with CAPTURE_LOCK.shared():
            # Rows still running at the cap keep everything they produced.
            length = torch.where(self.finished, self.length, self.frame.expand(B)).tolist()
            return [self.codes[:length[b], b].clone() for b in range(B)]
