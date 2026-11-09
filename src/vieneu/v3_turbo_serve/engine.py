"""
Batched serving engine for v3 Turbo (static batching).

Ties the batched backbone + batched acoustic frame generation into one loop that
advances B requests together:

    prefill(prompts) -> loop { acoustic frame (B) -> EOS check -> backbone step (B) }

Finished requests are masked out (their last frame is kept) and the batch keeps
running until all finish or ``max_new_frames``. Each request's codes are decoded
to a waveform at the end. Pure PyTorch, runs on CUDA (or CPU).
"""
from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch

from .batched_acoustic import generate_frame_batched
from .batched_backbone import BatchedBackbone
from .._v3_turbo_engine.capture_lock import CAPTURE_LOCK
from .._v3_turbo_engine.rep_history import DEFAULT_REP_WINDOW, RepetitionHistory

logger = logging.getLogger("Vieneu.V3TurboServe")

# Guard chống "chunk câm": một row khỏe sinh ≥ ~0.4 frame trên mỗi ký tự phoneme
# (đo thực nghiệm 0.41–1.0); dưới 0.25 gần như chắc chắn là EOS sớm / cụt giữa
# chừng → sinh lại row đó. Độ dài phoneme được tính SAU khi loại các tag markup
# (<en>…</en>, <|emotion_k|>) vì chúng chiếm ký tự nhưng không tốn frame audio.
MIN_FRAMES_PER_PHONE = 0.25
_MARKUP_RE = re.compile(r"<\|emotion_\d+\|>|</?en>")


def _min_expected_frames(phonemes: str) -> int:
    eff_len = len(_MARKUP_RE.sub("", phonemes or ""))
    return max(3, math.ceil(MIN_FRAMES_PER_PHONE * eff_len))


# Chặn-trên đối xứng: trần frame theo độ dài phoneme (24 + 2.0/ký tự, đo trên
# dataset finetune — xem vieneu_utils.core_utils.max_expected_frames). Row ngắn
# bắn trượt stop token thì bị cắt tại trần của CHÍNH row đó thay vì chạy hết
# max_new_frames của cả batch.
from vieneu_utils.core_utils import max_expected_frames, BABBLE_MAX_RETRIES, babble_suspect, babble_prefer, check_sampling


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class V3TurboBatchEngine:
    def __init__(self, tts):
        # tts: VieNeuTTSv3Turbo (provides prompt building, embeddings, heads, codec)
        self.tts = tts
        self.model = tts.model
        self.config = tts.config
        self.bb = BatchedBackbone(tts.model)
        self._graphs = {}  # (B, temp, top_k, top_p) -> CudaGraphedFrame (acoustic step)
        # Whole-frame CUDA graphs (see fused.py), keyed by batch bucket, cache
        # length bucket and sampling settings. The default on CUDA;
        # VIENEU_FUSED_FRAME=0 falls back to the plain per-op loop.
        self._fused = {}
        self.use_fused = os.environ.get("VIENEU_FUSED_FRAME", "1") != "0"
        # Every run overwrites its graph's buffers (static KV cache, codes, flags),
        # so runs must not overlap — infer() is called from several threads by the
        # Gradio app and by servers. Lock order: _fused_lock, then CAPTURE_LOCK
        # (process-wide: a capture keeps all other v3 Turbo GPU work out, see
        # _v3_turbo_engine/capture_lock.py) — never the other way round.
        self._fused_lock = threading.Lock()
        # Each graph pins its own static KV cache (hundreds of MB at batch 32), and
        # a new sampling setting (e.g. a temperature slider) means new graphs: only
        # the most recently used ``fused_samplings`` settings keep theirs.
        self._fused_lru: "OrderedDict[tuple, None]" = OrderedDict()
        self.fused_samplings = max(1, _env_int("VIENEU_FUSED_SAMPLINGS", 2))
        # Called once per generated frame (both loops). A server sets it to
        # its cancel check; raising from it stops the batch within a frame.
        self.frame_hook = None

    def _get_graph(self, B, temperature, top_k, top_p):
        key = (B, round(temperature, 4), top_k, round(top_p, 4))
        if key not in self._graphs:
            from .cudagraph import CudaGraphedFrame
            with CAPTURE_LOCK.exclusive():
                if key not in self._graphs:
                    self._graphs[key] = CudaGraphedFrame(
                        self.model, B, temperature=temperature, top_k=top_k, top_p=top_p
                    )
        return self._graphs[key]

    @torch.no_grad()
    def _prompt_embeds(self, req) -> torch.Tensor:
        # Each request carries its own reference (speaker_emb + ref_codes) and optional
        # use_ref_codes switch. The speaker anchor is added to every prefill row.
        # A "style" key in the request is accepted but ignored (deprecated): the style
        # is implied by the reference, so the head token is always the natural style.
        style_id = self.tts._resolve_style_id()
        phonemes = req.get("phonemes")
        if phonemes is None:
            from vieneu_utils.phonemize_text import phonemize_text_with_emotions
            phonemes = phonemize_text_with_emotions(req["text"])
        ref_codes = req.get("ref_codes") if req.get("use_ref_codes", True) else None
        spk = self.tts._resolve_speaker_emb(req.get("speaker_emb"))
        prompt_2d = self.tts._build_prompt_2d(phonemes, None, ref_codes, style_id)
        return self.model._build_inputs_embeds(prompt_2d.unsqueeze(0).to(self.tts.device), speaker_emb=spk)[0]  # (T, H)

    @torch.no_grad()
    def generate_batch(
        self,
        requests: List[dict],
        *,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_new_frames: int = 300,
        use_cudagraph: bool = False,
        max_retries: int = 2,
        frame_cap: bool = True,
    ) -> List[np.ndarray]:
        """Generate a waveform for each request in one batched run.

        ``temperature`` is a scalar applied to every codebook. ``repetition_penalty``
        (default 1.2, matching the single-path engine) down-weights codes already
        produced per row/codebook to avoid repetition artifacts.

        Rows that come back suspiciously short (fewer than ``MIN_FRAMES_PER_PHONE``
        frames per phoneme char — an early-EOS "silent chunk") are regenerated as a
        smaller batch, up to ``max_retries`` times, keeping the longest attempt.
        Set ``max_retries=0`` to disable the guard.

        ``frame_cap=True`` (default) additionally caps each row at its own
        ``max_expected_frames(phonemes)`` — the upper-bound twin of the guard
        above: a short row that misses its stop token gets truncated at a
        length plausible for its text instead of babbling to ``max_new_frames``.

        ``use_cudagraph=True`` captures (and caches) a CUDA graph of the per-frame
        acoustic step for this batch size — big per-step speedup, reused across calls.
        It is ignored when ``repetition_penalty != 1.0`` (the penalty needs dynamic
        per-row history, which a static graph cannot hold).
        """
        # Before any GPU work: a NaN baked into a graph is a device-side assert.
        temperature, top_k, top_p, repetition_penalty = check_sampling(
            temperature, top_k, top_p, repetition_penalty)
        # Fill in phonemes up front so the guard can size-check every row (and so
        # retries don't re-phonemize).
        reqs = []
        for r in requests:
            if r.get("phonemes") is None:
                from vieneu_utils.phonemize_text import phonemize_text_with_emotions
                r = {**r, "phonemes": phonemize_text_with_emotions(r["text"])}
            reqs.append(r)

        sampling = dict(
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, repetition_window=repetition_window,
            max_new_frames=max_new_frames, use_cudagraph=use_cudagraph,
        )
        # Trần frame RIÊNG từng row (luôn > floor chặn-dưới nên không kích retry).
        caps = ([min(max_new_frames, max_expected_frames(r["phonemes"])) for r in reqs]
                if frame_cap else None)
        codes = self._generate_codes_batch(reqs, frame_caps=caps, **sampling)

        # Guard: re-run rows whose output is too short for their text.
        floors = [_min_expected_frames(r["phonemes"]) for r in reqs]
        for _ in range(max(0, max_retries)):
            bad = [i for i in range(len(reqs)) if len(codes[i]) < floors[i]]
            if not bad:
                break
            logger.warning(
                f"⚠️ v3 Turbo batch: {len(bad)} chunk ngắn bất thường "
                f"(rows {bad}) — đang sinh lại...")
            retry = self._generate_codes_batch(
                [reqs[i] for i in bad],
                frame_caps=[caps[i] for i in bad] if caps is not None else None,
                **sampling)
            for j, i in enumerate(bad):
                if len(retry[j]) > len(codes[i]):
                    codes[i] = retry[j]
        else:
            still_bad = [i for i in range(len(reqs)) if len(codes[i]) < floors[i]]
            if still_bad and max_retries > 0:
                logger.warning(
                    f"⚠️ v3 Turbo batch: rows {still_bad} vẫn ngắn hơn ngưỡng sau "
                    f"{max_retries} lần thử — giữ bản dài nhất.")

        def _decode(c) -> np.ndarray:
            if not len(c):
                return np.zeros(0, dtype=np.float32)
            with CAPTURE_LOCK.shared():   # the codec runs on the GPU too
                return self.tts._decode_codes(c.cpu())

        wavs: List[np.ndarray] = [_decode(c) for c in codes]

        # Babble guard (đối xứng với guard "chunk câm" ở trên): row rất ngắn mà
        # "nói thêm" sau stop token trượt -> sinh lại row đó. Đây là tầng engine
        # nên mọi lối vào (SDK, Gradio, server) đều được che. Xem core_utils.babble_suspect.
        retries = int(getattr(self, "babble_retries", BABBLE_MAX_RETRIES))
        if retries > 0 and caps is not None:
            sr = int(getattr(self.tts, "sample_rate", 48_000))
            state = [babble_suspect(wavs[i], sr, reqs[i]["phonemes"], caps[i], len(codes[i]))
                     for i in range(len(reqs))]
            for attempt in range(retries):
                bad = [i for i in range(len(reqs)) if state[i][0]]
                if not bad:
                    break
                logger.info(f"🔁 v3 Turbo batch: {len(bad)} chunk ngắn nghi 'nói thêm' (rows {bad}) "
                            f"— đang sinh lại ({attempt + 1}/{retries})...")
                retry = self._generate_codes_batch(
                    [reqs[i] for i in bad], frame_caps=[caps[i] for i in bad], **sampling)
                for j, i in enumerate(bad):
                    w2 = _decode(retry[j])
                    cand = babble_suspect(w2, sr, reqs[i]["phonemes"], caps[i], len(retry[j]))
                    if babble_prefer(cand, state[i]):
                        codes[i], wavs[i], state[i] = retry[j], w2, cand
        return wavs

    @torch.no_grad()
    def _generate_codes_batch(
        self,
        requests: List[dict],
        *,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_new_frames: int = 300,
        use_cudagraph: bool = False,
        frame_caps: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """One batched generation pass; returns per-row codes ``(T, n_vq)`` (no decode).

        ``frame_caps[b]`` (nếu có) là trần frame riêng của row ``b`` — chạm trần
        thì row đó coi như xong (mask ra khỏi batch) dù chưa thấy EOS.
        """
        cfg = self.config
        n_vq = cfg.n_vq
        eos_id = cfg.speech_generation_end_token_id
        sgs = cfg.speech_generation_start_token_id
        pad = cfg.audio_pad_token_id
        dev = self.tts.device

        # Prompt embeddings and speaker anchors are built on the device: no
        # capture may run meanwhile. Released before the fused path, which takes
        # _fused_lock (see the lock order in __init__).
        with CAPTURE_LOCK.shared():
            embeds_list = [self._prompt_embeds(r) for r in requests]

            # Per-request speaker anchor, stacked to (B, D), re-added at every decode step
            # (broadcast over the single new row). None when the model has no speaker encoder.
            spk_list = [self.tts._resolve_speaker_emb(r.get("speaker_emb")) for r in requests]
            batch_spk = torch.cat(spk_list, dim=0) if (spk_list and spk_list[0] is not None) else None
        B = len(requests)

        if self.use_fused and dev.type == "cuda":
            return self._generate_codes_fused(
                embeds_list, batch_spk, frame_caps,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty, repetition_window=repetition_window,
                max_new_frames=max_new_frames,
            )

        # Per-row, per-codebook sliding-window history for the repetition penalty
        # (matches the single-path decode_one_frame). None when the penalty is disabled.
        history = ([RepetitionHistory(n_vq, repetition_window) for _ in range(B)]
                   if not math.isclose(repetition_penalty, 1.0) else None)
        # CUDA graph bakes in a static step → incompatible with dynamic rep-penalty.
        use_graph = use_cudagraph and dev.type == "cuda" and math.isclose(repetition_penalty, 1.0)
        graphed = self._get_graph(B, temperature, top_k, top_p) if use_graph else None

        # The per-op loop is GPU work, not a capture: it holds CAPTURE_LOCK shared
        # frame by frame (the hook outside), so a capture elsewhere waits for one
        # frame rather than the whole batch.
        with CAPTURE_LOCK.shared():
            h, cache, mask, pos = self.bb.prefill(embeds_list)   # h: (B, H)
        finished = [False] * B
        codes_per_req: List[List[torch.Tensor]] = [[] for _ in range(B)]

        for _ in range(max_new_frames):
            if self.frame_hook is not None:
                self.frame_hook()
            with CAPTURE_LOCK.shared():
                if graphed is not None:
                    codes, is_eos = graphed.run(h)               # acoustic frame via CUDA graph
                else:
                    codes, prefill_out = generate_frame_batched(
                        self.model, h, temperature=temperature, top_k=top_k, top_p=top_p,
                        repetition_penalty=repetition_penalty, history=history,
                    )
                    is_eos = (self.model.text_lm_head(prefill_out[:, 0]).float().argmax(-1) == eos_id)
                for b in range(B):
                    if not finished[b]:
                        codes_per_req[b].append(codes[b])   # include the EOS frame (matches single path)
                        if bool(is_eos[b]) or (
                            frame_caps is not None and len(codes_per_req[b]) >= frame_caps[b]
                        ):
                            finished[b] = True
                if all(finished):
                    break

                # Feed the generated frame back as the next backbone input.
                slot = torch.full((B, 1, n_vq + 1), pad, dtype=torch.long, device=dev)
                slot[:, :, 0] = sgs
                slot[:, 0, 1:] = codes
                se = self.model._build_inputs_embeds(slot, speaker_emb=batch_spk)
                h, cache, mask, pos = self.bb.decode_step(se, cache, mask, pos)

        with CAPTURE_LOCK.shared():
            return [
                torch.stack(codes_per_req[b]) if codes_per_req[b]
                else torch.zeros(0, cfg.n_vq, dtype=torch.long)
                for b in range(B)
            ]

    # Frames a fused graph's code buffer holds; a call may ask for fewer.
    FUSED_MAX_FRAMES = 512

    def warm_fused(self, batch_sizes=(1, 16), max_len: int = 1024, *,
                   temperature: float = 0.8, top_k: int = 25, top_p: float = 0.95,
                   repetition_penalty: float = 1.2,
                   repetition_window: int = DEFAULT_REP_WINDOW) -> int:
        """Capture the fused graphs a server will need before its first request.

        A capture costs ~0.5 s of warm-up frames, paid on the first call for
        each (batch bucket, cache length, sampling) otherwise. ``max_len`` is
        the cache-length bucket (prompt + frames; 1024 covers a 256-char chunk
        with a 30 s reference). Returns how many graphs were captured.
        """
        from .fused import FusedFrame, batch_bucket, cache_len_bucket

        if not (self.use_fused and self.tts.device.type == "cuda"):
            return 0
        max_len = cache_len_bucket(max_len, int(self.config.max_position_embeddings))
        n = 0
        CAPTURE_LOCK.ensure_not_held("warm_fused")   # lock order: _fused_lock first
        with self._fused_lock:
            for b in batch_sizes:
                bp = batch_bucket(int(b))
                key = (bp, max_len, round(temperature, 4), int(top_k), round(top_p, 4),
                       round(repetition_penalty, 4), int(repetition_window))
                if key not in self._fused:
                    n += 1
                self._fused_frame(key, lambda bp=bp: FusedFrame(
                    self.model, bp, max_len, self.FUSED_MAX_FRAMES,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty, repetition_window=repetition_window,
                ))
        return n

    def _fused_frame(self, key: tuple, create):
        """The cached graph for ``key``; ``create()`` captures it on first use.

        Call with ``_fused_lock`` held. Using a sampling setting (``key[2:]``)
        makes it the most recent; the graphs of older settings beyond
        ``fused_samplings`` are dropped before a new capture allocates.
        Dropping and capturing hold ``CAPTURE_LOCK`` exclusively: no other v3
        Turbo GPU work, from any thread or engine, runs meanwhile.
        """
        sampling = key[2:]
        self._fused_lru[sampling] = None
        self._fused_lru.move_to_end(sampling)
        fused = self._fused.get(key)
        if fused is not None and len(self._fused_lru) <= self.fused_samplings:
            return fused
        with CAPTURE_LOCK.exclusive():
            while len(self._fused_lru) > self.fused_samplings:
                old, _ = self._fused_lru.popitem(last=False)
                for k in [k for k in self._fused if k[2:] == old]:
                    del self._fused[k]
                logger.info("v3 Turbo: freed the CUDA graphs of sampling %s (keeping %d settings; "
                            "VIENEU_FUSED_SAMPLINGS)", old, self.fused_samplings)
            fused = self._fused.get(key)
            if fused is None:
                fused = self._fused[key] = create()
        return fused

    @torch.no_grad()
    def _generate_codes_fused(
        self, embeds_list, batch_spk, frame_caps, *, temperature, top_k, top_p,
        repetition_penalty, repetition_window, max_new_frames,
    ) -> List[torch.Tensor]:
        """``_generate_codes_batch`` through one captured CUDA graph per frame.

        The batch is padded to a power of two (repeating the last row) and the
        KV cache length rounded up to a bucket, so a handful of graphs serve
        every call; a graph costs a few warm-up frames to capture and is kept.
        """
        from .fused import FusedFrame, batch_bucket, cache_len_bucket

        cfg = self.config
        B = len(embeds_list)
        Bp = batch_bucket(B)
        pad = Bp - B
        embeds_p = list(embeds_list) + [embeds_list[-1]] * pad
        max_new_frames = min(int(max_new_frames), self.FUSED_MAX_FRAMES)
        caps = [min(int(c), max_new_frames) for c in (frame_caps or [max_new_frames] * B)]
        caps += [caps[-1]] * pad
        T = max(e.shape[0] for e in embeds_p)
        max_len = cache_len_bucket(T + max_new_frames + 1, int(cfg.max_position_embeddings))
        key = (Bp, max_len, round(temperature, 4), int(top_k), round(top_p, 4),
               round(repetition_penalty, 4), int(repetition_window))
        # Lock order: _fused_lock first. A thread holding CAPTURE_LOCK here would
        # wait for a _fused_lock holder that may be waiting for it to capture.
        CAPTURE_LOCK.ensure_not_held("VieNeu batch generation (infer, infer_batch)")
        with self._fused_lock:
            fused = self._fused_frame(key, lambda: FusedFrame(
                self.model, Bp, max_len, self.FUSED_MAX_FRAMES,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty, repetition_window=repetition_window,
            ))
            with CAPTURE_LOCK.shared():
                spk_p = None
                if batch_spk is not None:
                    spk_p = torch.cat([batch_spk, batch_spk[-1:].expand(pad, -1)], dim=0) if pad else batch_spk
                h, cache, mask, pos = self.bb.prefill(embeds_p)
            # run() holds CAPTURE_LOCK frame by frame: a capture elsewhere waits
            # for one frame, not for the whole batch.
            codes = fused.run(h, cache, mask, pos, caps, spk_p, max_new_frames, on_frame=self.frame_hook)
        return codes[:B]   # run() returns clones, safe to use after the lock
