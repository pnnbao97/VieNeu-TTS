from pathlib import Path
from typing import Optional, Union, List, Generator, Any, Dict
import numpy as np
import torch
import gc
import logging
from collections import defaultdict
from .base import BaseVieneuTTS
from .utils import _compile_codec_with_triton, extract_speech_ids, _linear_overlap_add, normalize_device
from vieneu_utils.phonemize_text import phonemize_batch
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks

logger = logging.getLogger("Vieneu.Fast")

class FastVieNeuTTS(BaseVieneuTTS):
    """
    GPU-optimized VieNeu-TTS using LMDeploy TurbomindEngine.
    """

    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS",
        backbone_device: str = "cuda",
        codec_repo: str = "neuphonic/distill-neucodec",
        codec_device: str = "cuda",
        memory_util: float = 0.3,
        tp: int = 1,
        enable_prefix_caching: bool = True,
        quant_policy: int = 0,
        enable_triton: bool = True,
        max_batch_size: int = 4,
        hf_token: Optional[str] = None,
    ):
        super().__init__()
        self.device = backbone_device

        if backbone_device != "cuda" and not backbone_device.startswith("cuda:"):
            raise ValueError("LMDeploy backend requires CUDA device")

        # Streaming configuration
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 50
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        self.max_batch_size = max_batch_size
        self._ref_cache: Dict[str, Any] = {}
        self.stored_dict = defaultdict(dict)

        self._is_onnx_codec = False
        self._triton_enabled = False

        self._load_backbone_lmdeploy(backbone_repo, memory_util, tp, enable_prefix_caching, quant_policy, hf_token)
        self._load_codec(codec_repo, codec_device, enable_triton)
        self._load_voices(backbone_repo, hf_token)
        self._warmup_model()

        logger.info("✅ FastVieNeuTTS with optimizations loaded successfully!")
        logger.info(f"   Max batch size: {self.max_batch_size}")

    def _load_backbone_lmdeploy(self, repo, memory_util, tp, enable_prefix_caching, quant_policy, hf_token=None):
        logger.info(f"Loading backbone with LMDeploy from: {repo}")
        if hf_token:
            import os
            os.environ["HF_TOKEN"] = hf_token

        try:
            from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
        except ImportError as e:
            raise ImportError(
                "Failed to import `lmdeploy`. Install with: pip install vieneu[gpu]"
            ) from e

        backend_config = TurbomindEngineConfig(
            cache_max_entry_count=memory_util,
            tp=tp,
            enable_prefix_caching=enable_prefix_caching,
            dtype='bfloat16',
            quant_policy=quant_policy
        )
        self.backbone = pipeline(repo, backend_config=backend_config)
        self.gen_config = GenerationConfig(
            top_p=0.95, top_k=50, temperature=1.0, max_new_tokens=2048,
            repetition_penalty=1.1,
            do_sample=True, min_new_tokens=40,
        )

    def _load_codec(self, codec_repo: str, codec_device: str, enable_triton: bool) -> None:
        super()._load_codec(codec_repo, codec_device)

        if enable_triton and not getattr(self, "_is_onnx_codec", False) and codec_device != "cpu":
            self._triton_enabled = _compile_codec_with_triton(self.codec)

    def _warmup_model(self):
        logger.info("🔥 Warming up model...")
        try:
            dummy_codes = list(range(10))
            dummy_prompt = self._format_prompt(dummy_codes, "warmup", "test")
            _ = self.backbone([dummy_prompt], gen_config=self.gen_config, do_preprocess=False)
            logger.info("   ✅ Warmup complete")
        except Exception as e:
            logger.warning(f"   ⚠️ Warmup failed: {e}")

    def _decode(self, codes_str: str) -> np.ndarray:
        speech_ids = extract_speech_ids(codes_str)
        if not speech_ids:
            raise ValueError(
                "No valid speech tokens found in the output. "
                "Lỗi này có thể do GPU của bạn không hỗ trợ định dạng bfloat16 (ví dụ: dòng T4, RTX 20-series) "
                "dẫn đến sai số khi tính toán. Bạn hãy thử chuyển sang dùng phiên bản VieNeu-TTS-0.3B nếu vẫn muốn dùng LmDeploy hoặc "
                "bỏ chọn 'LMDeploy' trong Tùy chọn nâng cao. Nếu vẫn gặp lỗi này, hãy thông báo với chúng tôi tại: https://discord.com/invite/yJt8kzjzWZ"
            )

        if self._is_onnx_codec:
            codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            recon = self.codec.decode_code(codes)
        else:
            with torch.no_grad():
                codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(self.codec.device)
                recon = self.codec.decode_code(codes).cpu().numpy()
        return recon[0, 0, :]


    def infer(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, max_chars: int = 256, silence_p: float = 0.15, crossfade_p: float = 0.0, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False, apply_watermark: bool = True) -> np.ndarray:

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        self.gen_config.temperature = temperature
        self.gen_config.top_k = top_k

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        if len(chunks) == 1:
            prompt = self._format_prompt(ref_codes, ref_text, chunks[0])
            responses = self.backbone([prompt], gen_config=self.gen_config, do_preprocess=False)
            wav = self._decode(responses[0].text)
            if apply_watermark:
                wav = self._apply_watermark(wav)
        else:
            all_wavs = self.infer_batch(chunks, ref_codes, ref_text, voice=voice, temperature=temperature, top_k=top_k, skip_normalize=True, apply_watermark=False)
            wav = join_audio_chunks(all_wavs, self.sample_rate, silence_p, crossfade_p)
            if apply_watermark:
                wav = self._apply_watermark(wav)

        return wav

    def infer_batch(self, texts: List[str], ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False, apply_watermark: bool = True, max_batch_size: Optional[int] = None) -> List[np.ndarray]:

        if not skip_normalize:
            texts = [self.normalizer.normalize(t) for t in texts]

        max_batch_size = max_batch_size or self.max_batch_size

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        # Pre-phonemize all for performance
        ref_phonemes = self.get_ref_phonemes(ref_text)
        chunk_phonemes = phonemize_batch(texts, skip_normalize=True)

        self.gen_config.temperature = temperature
        self.gen_config.top_k = top_k

        all_wavs = []
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i : i + max_batch_size]
            batch_phonemes = chunk_phonemes[i : i + max_batch_size]
            prompts = [self._format_prompt(ref_codes, ref_text, text, ref_phonemes=ref_phonemes, input_phonemes=ph)
                      for text, ph in zip(batch_texts, batch_phonemes)]
            responses = self.backbone(prompts, gen_config=self.gen_config, do_preprocess=False)
            batch_codes = [response.text for response in responses]
            batch_wavs = [self._decode(codes) for codes in batch_codes]
            if apply_watermark:
                batch_wavs = [self._apply_watermark(w) for w in batch_wavs]
            all_wavs.extend(batch_wavs)
        return all_wavs

    def infer_stream(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, max_chars: int = 256, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False) -> Generator[np.ndarray, None, None]:

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        self.gen_config.temperature = temperature
        self.gen_config.top_k = top_k

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        for chunk in chunks:
            yield from self._infer_stream_single(chunk, ref_codes, ref_text)

    def _infer_stream_single(self, text: str, ref_codes: Any, ref_text: str) -> Generator[np.ndarray, None, None]:
        ref_codes_list = self.to_list(ref_codes)
        prompt = self._format_prompt(ref_codes_list, ref_text, text)
        audio_cache = []
        token_cache = [f"<|speech_{idx}|>" for idx in ref_codes_list]
        n_decoded_samples = 0
        n_decoded_tokens = len(ref_codes_list)

        for response in self.backbone.stream_infer([prompt], gen_config=self.gen_config, do_preprocess=False):
            output_str = response.text
            new_tokens = output_str[len("".join(token_cache[len(ref_codes_list):])):] if len(token_cache) > len(ref_codes_list) else output_str
            if new_tokens:
                token_cache.append(new_tokens)

            if len(token_cache[n_decoded_tokens:]) >= self.streaming_frames_per_chunk + self.streaming_lookforward:
                tokens_start = max(n_decoded_tokens - self.streaming_lookback - self.streaming_overlap_frames, 0)
                tokens_end = n_decoded_tokens + self.streaming_frames_per_chunk + self.streaming_lookforward + self.streaming_overlap_frames
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = sample_start + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length
                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = self._apply_watermark(recon)
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)

                processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[n_decoded_samples:new_samples_end]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk
                yield processed_recon

        remaining_tokens = len(token_cache) - n_decoded_tokens
        if remaining_tokens > 0:
            tokens_start = max(len(token_cache) - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens), 0)
            sample_start = (len(token_cache) - tokens_start - remaining_tokens - self.streaming_overlap_frames) * self.hop_length
            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = self._apply_watermark(recon)
            recon = recon[sample_start:]
            audio_cache.append(recon)
            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]
            yield processed_recon

    def cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def get_optimization_stats(self) -> Dict[str, Any]:
        return {
            'triton_enabled': self._triton_enabled,
            'max_batch_size': self.max_batch_size,
            'cached_references': len(self._ref_cache),
            'active_sessions': len(self.stored_dict),
            'prefix_caching': True,
        }
