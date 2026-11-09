"""
VieNeu-TTS v3 Turbo — inference engine (PyTorch).
=================================================
Synthesizes 48 kHz Vietnamese / English speech with instant voice cloning from a
short reference clip. It loads a checkpoint plus the MOSS audio codec (and, for
cloning, an ONNX speaker encoder + denoiser) and turns text into a waveform.

Quick start:
    from vieneu._v3_turbo_engine import VieNeuTTSv3Turbo
    tts = VieNeuTTSv3Turbo()
    spk, codes = tts.prepare_reference("reference_voice.wav")   # enroll a voice once
    wav = tts.infer(phonemes=ph, speaker_emb=spk, ref_codes=codes)

Credits
-------
- VieNeu-TTS v3 Turbo is designed and trained from scratch on ~10,000 hours of
  English–Vietnamese speech by Phạm Nguyễn Ngọc Bảo — https://github.com/pnnbao97
- Phonemizer: sea-g2p — https://github.com/pnnbao97/sea-g2p
- Audio codec: MOSS-Audio-Tokenizer-Nano (OpenMOSS-Team).
"""
from __future__ import annotations
import math
import threading
from pathlib import Path
import time
from typing import Generator, List, Optional, Tuple, Union
import numpy as np
import torch
_STREAM_LEADIN_FRAMES = 4
from .configuration_v3_turbo import VieNeuV3TurboConfig
from .hub_load_v3_turbo import load_v3_turbo_checkpoint
from .modeling_v3_turbo import VieNeuV3TurboForTTS, _sample_token
from .capture_lock import CAPTURE_LOCK, gpu_work
from .rep_history import DEFAULT_REP_WINDOW, RepetitionHistory
from vieneu_utils.core_utils import (BABBLE_MAX_RETRIES, CODEC_SAMPLES_PER_FRAME, babble_suspect,
                                     babble_prefer, babble_log_line)
import logging

# Reference clips longer than this are trimmed before enrollment.
_MAX_REF_SECONDS = 8.0


class VieNeuTTSv3Turbo:
    """High-level text-to-speech interface (48 kHz, instant voice cloning).

    Enroll a voice once with :meth:`prepare_reference` (which denoises the clip,
    then extracts a speaker embedding and reference codes), then call :meth:`infer`
    per phonemized chunk. Calls are thread-safe (guarded by an internal lock).
    """

    SAMPLE_RATE = 48000

    def __init__(self, checkpoint_path: str='pnnbao-ump/VieNeu-TTS-v3-Turbo', model_subfolder: Optional[str]='update', tokenizer_path: Optional[str]=None, moss_tokenizer_path: str='OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano', denoiser_filename: str='denoiser.onnx', device: str='auto', dtype: str='auto'):
        """Load the v3 Turbo checkpoint, the MOSS codec, and (for cloning) the
        ONNX speaker encoder + denoiser.

        Args:
            checkpoint_path: HF repo id or local directory of the v3 Turbo model.
            model_subfolder: subfolder inside the repo holding the weights/tokenizer
                (the speaker encoder + denoiser are read from the repo root).
            device: ``"auto"`` | ``"cpu"`` | ``"cuda"``.
            dtype: ``"auto"`` | ``"float32"`` | ``"bfloat16"`` | ``"float16"``.
        """
        self._lock = threading.RLock()
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        from transformers import AutoTokenizer, AutoModel
        tok_path = tokenizer_path or checkpoint_path
        tok_sub = None if tokenizer_path else (model_subfolder or None)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, subfolder=tok_sub or "", trust_remote_code=True)
        self.config = VieNeuV3TurboConfig.from_pretrained(checkpoint_path, subfolder=model_subfolder or "")
        # Weights live in a subfolder, so the Hub's download counter (which keys on the
        # root config.json) wouldn't register the load — touch the root file once so each
        # load counts like a normal download. Best-effort, never fatal.
        if model_subfolder:
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(checkpoint_path, "config.json")
            except Exception:
                pass
        self.model = load_v3_turbo_checkpoint(checkpoint_path, device=self.device, dtype=self.dtype, subfolder=model_subfolder).eval()
        self.audio_tokenizer = AutoModel.from_pretrained(moss_tokenizer_path, trust_remote_code=True).to(self.device).eval()

        self.default_style = "tu_nhien"

        # Speaker encoder + denoiser (voice cloning). Both are ONNX and read from the
        # repo root; loaded only when the checkpoint uses a speaker embedding.
        self.speaker_encoder = None
        self.denoiser = None
        self.use_speaker_embedding = bool(getattr(self.config, "use_speaker_embedding", False))
        if self.use_speaker_embedding:
            from .speaker import OnnxSpeakerEncoder
            self.speaker_encoder = OnnxSpeakerEncoder.from_pretrained(
                checkpoint_path, filename=getattr(self.config, "speaker_encoder_filename", "speaker_encoder.onnx"),
                device="cpu",
            )
            if denoiser_filename:
                try:
                    from huggingface_hub import hf_hub_download
                    from .onnx_denoiser import OnnxDenoiser
                    _local = Path(checkpoint_path) / denoiser_filename
                    dn_path = str(_local) if _local.is_file() else hf_hub_download(checkpoint_path, denoiser_filename)
                    self.denoiser = OnnxDenoiser(dn_path)
                except Exception:
                    self.denoiser = None

    # ── Style / speaker resolution ─────────────────────────────────────────────

    def _resolve_style_id(self, style=None) -> int:
        """Always the natural-style token (DEPRECATED ``style``, kept for compat).

        The speaking style is carried by the reference (speaker embedding + ref
        codes), so the head token is fixed to the natural style; whatever ``style``
        a caller passes is ignored.
        """
        return int(self.config.default_style_token_id)

    @gpu_work
    def _resolve_speaker_emb(self, speaker_emb: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        if not self.use_speaker_embedding:
            return None
        if speaker_emb is None:
            raise ValueError("This model needs a speaker anchor: pass `speaker_emb` (see prepare_reference).")
        arr = np.asarray(speaker_emb, dtype=np.float32).reshape(-1)
        if not arr.any():
            raise ValueError("speaker_emb is all-zero — not a valid speaker anchor.")
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)

    # ── Reference enrollment ────────────────────────────────────────────────────

    def extract_speaker_emb(self, ref_audio: Union[str, "torch.Tensor"], sr: Optional[int]=None) -> np.ndarray:
        """Return the 192-d speaker embedding for a reference clip (path or waveform)."""
        if self.speaker_encoder is None:
            raise RuntimeError("Model was not built with a speaker encoder.")
        wav, sr = self._load_mono(ref_audio, sr)
        return self.speaker_encoder.embed(wav, sr)

    def prepare_reference(self, ref_audio: Union[str, "torch.Tensor"], *, sr: Optional[int]=None, denoise: bool=True, use_ref_codes: bool=True, max_seconds: float=_MAX_REF_SECONDS) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Enroll a voice: trim to ``max_seconds``, optionally denoise, then return
        ``(speaker_emb, ref_codes)``. Compute once and reuse for every chunk.
        """
        wav, sr = self._load_mono(ref_audio, sr)
        if wav.shape[-1] > int(max_seconds * sr):
            wav = wav[..., : int(max_seconds * sr)]
        if denoise and self.denoiser is not None:
            clean = self.denoiser.denoise(wav[0].cpu().numpy(), sr)   # -> float32 @ 44100
            wav = torch.from_numpy(clean).unsqueeze(0)
            sr = 44100
        speaker_emb = self.speaker_encoder.embed(wav, sr) if self.speaker_encoder is not None else None
        ref_codes = self._encode_ref_wav(wav, sr) if use_ref_codes else None
        return speaker_emb, ref_codes

    # ── Public synthesis ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_phonemes(phonemes: Optional[str], text: Optional[str]) -> str:
        """Chuỗi phoneme cuối cùng đưa vào prompt (phonemize ``text`` nếu cần)."""
        if phonemes is not None:
            return phonemes
        from vieneu_utils.phonemize_text import phonemize_text_with_emotions
        return phonemize_text_with_emotions(text or "")

    def infer(self, phonemes: Optional[str]=None, text: Optional[str]=None, ref_codes: Optional[np.ndarray]=None, speaker_emb: Optional[np.ndarray]=None, style=None, use_ref_codes: bool=True, temperature: float=0.8, top_k: int=25, top_p: float=0.95, max_new_frames: int=300, repetition_penalty: float=1.2, repetition_window: int=DEFAULT_REP_WINDOW, frame_cap: bool=True) -> np.ndarray:
        """Synthesize one (already phonemized) chunk into a float32, 48 kHz waveform.

        Args:
            phonemes: SEA-G2P phoneme string. If ``None``, ``text`` is phonemized.
            ref_codes / speaker_emb: reference voice from :meth:`prepare_reference`.
            style: DEPRECATED and ignored — the speaking style is carried by the
                reference, so generation is always the natural style.
            use_ref_codes: keep the in-context reference frames (fidelity) or drop
                them and rely on the speaker embedding only (consistency).
            frame_cap: chặn ``max_new_frames`` theo độ dài phoneme
                (:func:`vieneu_utils.core_utils.max_expected_frames`) — chunk ngắn
                bắn trượt stop token thì phần "nói thêm" bị cắt cụt. ``False`` để
                dùng nguyên ``max_new_frames``.
        """
        phonemes = self._resolve_phonemes(phonemes, text)
        if frame_cap:
            from vieneu_utils.core_utils import max_expected_frames
            max_new_frames = min(max_new_frames, max_expected_frames(phonemes))
        gen = lambda: self._generate_codes(phonemes, None, ref_codes, speaker_emb, style, use_ref_codes, temperature, top_k, top_p, max_new_frames, repetition_penalty, repetition_window)
        codes = gen()
        wav = self._decode_codes(codes)
        # Babble guard: chunk rất ngắn "nói thêm" sau stop token trượt -> sinh lại
        # (xem vieneu_utils.core_utils.babble_suspect). Nằm ở tầng engine nên mọi
        # lối vào (SDK, Gradio, server) đều được che, không chỉ VieNeu.infer().
        retries = int(getattr(self, "babble_retries", BABBLE_MAX_RETRIES))
        if retries > 0 and frame_cap:
            sr = int(getattr(self, "sample_rate", 48_000))
            best = babble_suspect(wav, sr, phonemes, max_new_frames, len(codes))
            tries = 0
            while best[0] and tries < retries:
                c2 = gen()
                w2 = self._decode_codes(c2)
                cand = babble_suspect(w2, sr, phonemes, max_new_frames, len(c2))
                if babble_prefer(cand, best):
                    codes, wav, best = c2, w2, cand
                tries += 1
            if tries:
                logging.getLogger("Vieneu.V3Turbo").info("🔁 " + babble_log_line(best, tries, max_new_frames))
        return wav

    def infer_stream(self, phonemes: Optional[str]=None, text: Optional[str]=None, ref_codes: Optional[np.ndarray]=None, speaker_emb: Optional[np.ndarray]=None, style=None, use_ref_codes: bool=True, temperature: float=0.8, top_k: int=25, top_p: float=0.95, max_new_frames: int=300, chunk_frames: int=25, repetition_penalty: float=1.2, repetition_window: int=DEFAULT_REP_WINDOW, frame_cap: bool=True) -> Generator[np.ndarray, None, None]:
        """Like :meth:`infer` but yields the waveform in chunks for low latency."""
        phonemes = self._resolve_phonemes(phonemes, text)
        if frame_cap:
            from vieneu_utils.core_utils import max_expected_frames
            max_new_frames = min(max_new_frames, max_expected_frames(phonemes))
        spk_t = self._resolve_speaker_emb(speaker_emb)
        if not use_ref_codes:
            ref_codes = None
        style_id = self._resolve_style_id()
        prompt_2d = self._build_prompt_2d(phonemes, None, ref_codes, style_id)
        # The stream before this one may hold _lock while it waits for a capture.
        CAPTURE_LOCK.ensure_not_held("infer_stream")
        with self._lock:
            yield from self._stream_generate(prompt_2d, spk_t, temperature, top_k, top_p, max_new_frames, chunk_frames, repetition_penalty=repetition_penalty, repetition_window=repetition_window)

    # ── Generation core ─────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_gen_slot_row(slot_row: torch.Tensor, frame_codes: Optional[torch.Tensor], sgs_id: int, audio_pad: int) -> None:
        slot_row[:, :, 0] = sgs_id
        if frame_codes is None:
            slot_row[:, :, 1:] = audio_pad
        else:
            slot_row[:, 0, 1:] = frame_codes.to(slot_row.device)

    @torch.no_grad()
    def _generate_codes(self, phonemes, text, ref_codes, speaker_emb, style, use_ref_codes, temperature, top_k, top_p, max_new_frames, repetition_penalty: float=1.2, repetition_window: int=DEFAULT_REP_WINDOW) -> torch.LongTensor:
        # Device work holds CAPTURE_LOCK shared frame by frame: a capture in
        # another thread waits for one frame, not for the whole utterance.
        style_id = self._resolve_style_id()   # `style` deprecated/ignored
        with CAPTURE_LOCK.shared():
            spk_t = self._resolve_speaker_emb(speaker_emb)
            if not use_ref_codes:
                ref_codes = None
            prompt_2d = self._build_prompt_2d(phonemes, text, ref_codes, style_id)
            input_2d = prompt_2d.unsqueeze(0).to(self.device)
            prefill_embeds = self.model._build_inputs_embeds(input_2d, speaker_emb=spk_t)
            prefill_out = self.model.semantic_backbone(inputs_embeds=prefill_embeds, use_cache=True, return_dict=True)
            past_kv = prefill_out.past_key_values
            h = prefill_out.last_hidden_state[:, -1]
        all_codes: List[torch.LongTensor] = []
        eos_id = self.config.speech_generation_end_token_id
        sgs_id = self.config.speech_generation_start_token_id
        n_vq = self.config.n_vq
        audio_pad = self.config.audio_pad_token_id
        hist = RepetitionHistory(n_vq, repetition_window) if not math.isclose(repetition_penalty, 1.0) else None
        for _ in range(max_new_frames):
            with CAPTURE_LOCK.shared():
                frame_codes, last_local_out = self.model.decode_one_frame(h, text_token_id=torch.tensor([sgs_id], device=self.device), temperature=temperature, top_k=top_k, audio_top_p=top_p, repetition_penalty=repetition_penalty, history_by_channel=hist)
                all_codes.append(frame_codes.cpu())
                text_logits = self.model.text_lm_head(last_local_out[0, 0]).float()
                if int(text_logits.argmax().item()) == eos_id:
                    break
                slot_row = torch.full((1, 1, n_vq + 1), audio_pad, dtype=torch.long, device=self.device)
                self._prepare_gen_slot_row(slot_row, frame_codes=frame_codes, sgs_id=sgs_id, audio_pad=audio_pad)
                slot_embed = self.model._build_inputs_embeds(slot_row, speaker_emb=spk_t)
                step_out = self.model.semantic_backbone(inputs_embeds=slot_embed, past_key_values=past_kv, use_cache=True, return_dict=True)
                past_kv = step_out.past_key_values
                h = step_out.last_hidden_state[:, 0]
        if not all_codes:
            return torch.zeros(0, self.config.n_vq, dtype=torch.long)
        return torch.stack(all_codes)

    @torch.no_grad()
    def _stream_generate(self, prompt_2d, spk_t, temperature, top_k, top_p, max_new_frames, chunk_frames, repetition_penalty: float=1.2, repetition_window: int=DEFAULT_REP_WINDOW) -> Generator[np.ndarray, None, None]:
        # Device work holds CAPTURE_LOCK shared step by step, never across a yield.
        with CAPTURE_LOCK.shared():
            input_2d = prompt_2d.unsqueeze(0).to(self.device)
            prefill_embeds = self.model._build_inputs_embeds(input_2d, speaker_emb=spk_t)
            prefill_out = self.model.semantic_backbone(inputs_embeds=prefill_embeds, use_cache=True, return_dict=True)
            past_kv = prefill_out.past_key_values
            h = prefill_out.last_hidden_state[:, -1]
        eos_id = self.config.speech_generation_end_token_id
        sgs_id = self.config.speech_generation_start_token_id
        n_vq = self.config.n_vq
        audio_pad = self.config.audio_pad_token_id
        buffer: List[torch.LongTensor] = []
        hist = RepetitionHistory(n_vq, repetition_window) if not math.isclose(repetition_penalty, 1.0) else None
        sr = self.SAMPLE_RATE
        first_decode = True
        emitted_samples = 0
        t_first: Optional[float] = None

        def _target_frames() -> int:
            cap = max(1, chunk_frames)
            if t_first is None:
                return min(cap, _STREAM_LEADIN_FRAMES)
            lead = emitted_samples / sr - (time.perf_counter() - t_first)
            if lead < 0.2:
                return min(cap, _STREAM_LEADIN_FRAMES)
            if lead < 0.55:
                return min(cap, 6)
            if lead < 1.1:
                return min(cap, 8)
            return cap
        try:
            for _ in range(max_new_frames):
                with CAPTURE_LOCK.shared():
                    frame_codes, last_local_out = self.model.decode_one_frame(h, text_token_id=torch.tensor([sgs_id], device=self.device), temperature=temperature, top_k=top_k, audio_top_p=top_p, repetition_penalty=repetition_penalty, history_by_channel=hist)
                    buffer.append(frame_codes.cpu())
                    text_logits = self.model.text_lm_head(last_local_out[0, 0]).float()
                    if int(text_logits.argmax().item()) == eos_id:
                        break
                    slot_row = torch.full((1, 1, n_vq + 1), audio_pad, dtype=torch.long, device=self.device)
                    self._prepare_gen_slot_row(slot_row, frame_codes=frame_codes, sgs_id=sgs_id, audio_pad=audio_pad)
                    slot_embed = self.model._build_inputs_embeds(slot_row, speaker_emb=spk_t)
                    step_out = self.model.semantic_backbone(inputs_embeds=slot_embed, past_key_values=past_kv, use_cache=True, return_dict=True)
                    past_kv = step_out.past_key_values
                    h = step_out.last_hidden_state[:, 0]
                if len(buffer) >= _target_frames():
                    wav = self._decode_codes_stream(torch.stack(buffer), reset=first_decode)
                    first_decode = False
                    if t_first is None and len(wav):
                        t_first = time.perf_counter()
                    emitted_samples += len(wav)
                    buffer = []
                    yield wav
            if buffer:
                wav = self._decode_codes_stream(torch.stack(buffer), reset=first_decode)
                first_decode = False
                yield wav
        finally:
            if not first_decode:
                self._reset_stream_session()

    # ── MOSS codec helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _moss_codes_to_Tnq(audio_codes: torch.Tensor, n_vq: int) -> torch.Tensor:
        if audio_codes.ndim != 3:
            raise ValueError(f'audio_codes must be 3D, got {tuple(audio_codes.shape)}')
        if audio_codes.shape[0] == n_vq:
            return audio_codes[:, 0, :].permute(1, 0)
        if audio_codes.shape[1] == n_vq:
            return audio_codes[0].permute(1, 0)
        raise ValueError(f'unexpected audio_codes shape {tuple(audio_codes.shape)} for n_vq={n_vq}')

    @staticmethod
    def _Tnq_to_moss_codes(codes_Tnq: torch.Tensor) -> torch.Tensor:
        return codes_Tnq.permute(1, 0).unsqueeze(1)

    def _load_mono(self, ref_audio: Union[str, "torch.Tensor"], sr: Optional[int]) -> Tuple[torch.Tensor, int]:
        """Return ``(wav (1, T) float32, sr)`` from a path or an in-memory waveform."""
        if isinstance(ref_audio, (str, bytes)) or hasattr(ref_audio, "__fspath__"):
            # soundfile is a base dependency; torchaudio is NOT part of the [cuda]
            # extra, so it is only a fallback for formats libsndfile cannot read.
            try:
                import soundfile as sf
                data, sr = sf.read(str(ref_audio), dtype="float32", always_2d=True)   # (T, C)
                wav = torch.from_numpy(np.ascontiguousarray(data.T))                    # (C, T)
            except Exception as e:  # noqa: BLE001
                try:
                    import torchaudio
                except ImportError:
                    raise RuntimeError(f"Không đọc được audio mẫu '{ref_audio}': {e}") from e
                wav, sr = torchaudio.load(str(ref_audio))
        else:
            wav = torch.as_tensor(ref_audio, dtype=torch.float32)
            if sr is None:
                raise ValueError("Pass `sr` when giving a waveform array.")
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return wav.float(), sr

    @staticmethod
    def _resample(wav: torch.Tensor, sr: int, target: int) -> torch.Tensor:
        """(C, T) → (C, T') via soxr (base dependency); torchaudio only as a fallback."""
        try:
            import soxr
            out = soxr.resample(wav.cpu().numpy().T, sr, target)         # (T, C) in / out
            return torch.from_numpy(np.ascontiguousarray(out.T.astype(np.float32)))
        except ImportError:
            import torchaudio
            return torchaudio.functional.resample(wav, sr, target)

    @gpu_work
    def _encode_ref_wav(self, wav: torch.Tensor, sr: int) -> np.ndarray:
        """Waveform ``(ch, n)`` → MOSS codes ``(T, n_vq)``, ``T = n_padded / 3840``.

        The clip is zero-padded to a whole number of codec frames first: fed an
        odd tail, the codec pads it itself and that frame always comes out as
        codebook-0 = 455, an audible blip rather than silence (issue #198). With
        real zeros the last frame encodes the actual tail (+ silence).
        """
        if sr != self.SAMPLE_RATE:
            wav = self._resample(wav, sr, self.SAMPLE_RATE)
        n = wav.shape[-1]
        if n % CODEC_SAMPLES_PER_FRAME:
            wav = torch.nn.functional.pad(wav, (0, CODEC_SAMPLES_PER_FRAME - n % CODEC_SAMPLES_PER_FRAME))
        n_frames = wav.shape[-1] // CODEC_SAMPLES_PER_FRAME
        n_ch = int(getattr(self.audio_tokenizer.config, 'number_channels', 2))
        wav = wav.repeat(n_ch, 1) if wav.shape[0] == 1 else wav[:n_ch]
        wav = wav.unsqueeze(0).to(self.device)
        with torch.no_grad():
            enc = self.audio_tokenizer.encode(wav, return_dict=True)
        codes = self._moss_codes_to_Tnq(enc.audio_codes, self.config.n_vq).cpu().numpy()
        return codes[:n_frames]

    def _encode_ref(self, ref_audio_path: str) -> np.ndarray:
        """Encode a reference wav into MOSS voice codes of shape ``(T, n_vq)``."""
        wav, sr = self._load_mono(ref_audio_path, None)
        return self._encode_ref_wav(wav, sr)

    @gpu_work
    @torch.no_grad()
    def _decode_codes(self, codes: torch.LongTensor) -> np.ndarray:
        if codes.numel() == 0:
            return np.zeros(0, dtype=np.float32)
        c = self._Tnq_to_moss_codes(codes).to(self.device)
        dec = self.audio_tokenizer.decode(c, return_dict=True)
        return dec.audio[0].mean(0).cpu().float().numpy()

    @gpu_work
    @torch.no_grad()
    def _decode_codes_stream(self, codes: torch.LongTensor, *, reset: bool) -> np.ndarray:
        c2d = codes.permute(1, 0).to(self.device)
        out = self.audio_tokenizer.batch_decode([c2d], num_quantizers=self.config.n_vq, streaming=True, reset_stream=reset)
        audio = out.audio
        if getattr(out, 'audio_lengths', None) is not None:
            audio = audio[:, :, : int(out.audio_lengths[0].item())]
        return audio[0].mean(0).cpu().float().numpy()

    @gpu_work
    def _reset_stream_session(self) -> None:
        reset = getattr(self.audio_tokenizer, '_reset_batch_decode_streaming_state', None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass

    # ── Prompt builder ──────────────────────────────────────────────────────────

    def _build_prompt_2d(self, phonemes: Optional[str], text: Optional[str], ref_codes: Optional[np.ndarray], style_token_id: int) -> torch.LongTensor:
        from .prompt_v3_turbo import build_prompt_2d
        if phonemes is not None:
            text_phones = phonemes
        else:
            from vieneu_utils.phonemize_text import phonemize_text_with_emotions
            text_phones = phonemize_text_with_emotions(text or "")
        ref_tensor: Optional[torch.LongTensor] = None
        if ref_codes is not None:
            ref_tensor = torch.as_tensor(ref_codes, dtype=torch.long)
        return build_prompt_2d(text_phones, ref_tensor, self.tokenizer, self.config, style_token_id=style_token_id)

    # ── Device / dtype ──────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        if dtype == 'auto':
            if self.device.type == 'cuda':
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        return {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[dtype]
