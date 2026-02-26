from typing import Generator, List, Dict, Any, Union, Optional
from pathlib import Path
import re
import numpy as np
import torch
import librosa
from collections import defaultdict
from neucodec import NeuCodec, DistillNeuCodec
from vieneu_utils.phonemize_text import phonemize_with_dict
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks
from .base import VieneuBase, _linear_overlap_add, _compile_codec_with_triton, SPEECH_TOKEN_RE

class FastVieNeuTTS(VieneuBase):
    """
    GPU-optimized VieNeu-TTS using LMDeploy TurbomindEngine.
    """

    def __init__(
        self,
        backbone_repo="pnnbao-ump/VieNeu-TTS",
        backbone_device="cuda",
        codec_repo="neuphonic/distill-neucodec",
        codec_device="cuda",
        memory_util=0.3,
        tp=1,
        enable_prefix_caching=True,
        quant_policy=0,
        enable_triton=True,
        max_batch_size=4,
        hf_token=None,
    ):
        if backbone_device != "cuda" and not backbone_device.startswith("cuda:"):
            raise ValueError("LMDeploy backend requires CUDA device")

        super().__init__()

        # Streaming settings
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 50
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        self.max_batch_size = max_batch_size
        self._ref_cache = {}
        self.stored_dict = defaultdict(dict)

        # Flags
        self._is_onnx_codec = False
        self._triton_enabled = False

        # Load models
        self._load_backbone(backbone_repo, memory_util, tp, enable_prefix_caching, quant_policy, hf_token)
        self._load_codec(codec_repo, codec_device, enable_triton)
        self._load_voices(backbone_repo, hf_token)
        self._warmup_model()

        print("✅ FastVieNeuTTS with optimizations loaded successfully!")

    def _load_backbone(self, repo, memory_util=0.3, tp=1, enable_prefix_caching=True, quant_policy=0, hf_token=None):
        print(f"Loading backbone with LMDeploy from: {repo}")
        if hf_token:
            import os
            os.environ["HF_TOKEN"] = hf_token

        from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
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
            do_sample=True, min_new_tokens=40,
        )

    def _load_codec(self, codec_repo, codec_device, enable_triton=True):
        print(f"Loading codec from: {codec_repo} on {codec_device}")
        match codec_repo:
            case "neuphonic/neucodec":
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/distill-neucodec":
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/neucodec-onnx-decoder-int8":
                if codec_device != "cpu": raise ValueError("ONNX decoder only runs on CPU")
                from neucodec import NeuCodecOnnxDecoder
                self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
                self._is_onnx_codec = True
            case _:
                raise ValueError(f"Unsupported codec repository: {codec_repo}")

        if enable_triton and not self._is_onnx_codec and codec_device != "cpu":
            self._triton_enabled = _compile_codec_with_triton(self.codec)

    def _warmup_model(self):
        print("🔥 Warming up model...")
        try:
            prompt = self._format_prompt([0]*10, "warmup", "test")
            self.backbone([prompt], gen_config=self.gen_config, do_preprocess=False)
        except Exception as e:
            print(f"   ⚠️ Warmup failed: {e}")

    def _decode(self, codes: str):
        speech_ids = [int(num) for num in SPEECH_TOKEN_RE.findall(codes)]
        if not speech_ids: raise ValueError("No valid speech tokens found.")

        if self._is_onnx_codec:
            recon = self.codec.decode_code(np.array(speech_ids, dtype=np.int32)[None, None, :])
        else:
            with torch.no_grad():
                codes_torch = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(self.codec.device)
                recon = self.codec.decode_code(codes_torch).cpu().numpy()
        return recon[0, 0, :]

    def _format_prompt(self, ref_codes: list, ref_text: str, input_text: str) -> str:
        rtp, itp = phonemize_with_dict(ref_text), phonemize_with_dict(input_text, skip_normalize=True)
        cs = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        return (f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{rtp} {itp}"
                f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{cs}")

    def infer(self, text: str, ref_audio=None, ref_codes=None, ref_text=None, max_chars=256, silence_p=0.15, crossfade_p=0.0, voice=None, temperature=1.0, top_k=50, skip_normalize=False) -> np.ndarray:
        if voice:
            ref_codes, ref_text = voice.get('codes', ref_codes), voice.get('text', ref_text)
        if ref_audio is not None and ref_codes is None:
            ref_codes = self.encode_reference(ref_audio)
        elif self._default_voice and (ref_codes is None or ref_text is None):
            v = self.get_preset_voice(None)
            ref_codes, ref_text = v['codes'], v['text']
        if ref_codes is None or ref_text is None:
             raise ValueError("Must provide either 'voice' dict or both 'ref_codes' and 'ref_text'.")

        if not skip_normalize: text = self.normalizer.normalize(text)
        self.gen_config.temperature, self.gen_config.top_k = temperature, top_k
        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks: return np.array([], dtype=np.float32)

        if len(chunks) == 1:
            if isinstance(ref_codes, (torch.Tensor, np.ndarray)): ref_codes = ref_codes.flatten().tolist()
            wav = self._decode(self.backbone([self._format_prompt(ref_codes, ref_text, chunks[0])], gen_config=self.gen_config, do_preprocess=False)[0].text)
        else:
            all_wavs = self.infer_batch(chunks, ref_codes, ref_text, temperature=temperature, top_k=top_k, skip_normalize=True)
            wav = join_audio_chunks(all_wavs, self.sample_rate, silence_p, crossfade_p)

        if self.watermarker: wav = self.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
        return wav

    def infer_batch(self, texts: list, ref_codes=None, ref_text=None, max_batch_size=None, voice=None, temperature=1.0, top_k=50, skip_normalize=False) -> list:
        if not skip_normalize: texts = [self.normalizer.normalize(t) for t in texts]
        max_batch_size = max_batch_size or self.max_batch_size
        if voice: ref_codes, ref_text = voice.get('codes', ref_codes), voice.get('text', ref_text)
        elif self._default_voice and (ref_codes is None or ref_text is None):
             v = self.get_preset_voice(None)
             ref_codes, ref_text = v['codes'], v['text']
        if ref_codes is None or ref_text is None: raise ValueError("Must provide voice info.")

        if isinstance(ref_codes, (torch.Tensor, np.ndarray)): ref_codes = ref_codes.flatten().tolist()
        self.gen_config.temperature, self.gen_config.top_k = temperature, top_k

        all_wavs = []
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i:i+max_batch_size]
            prompts = [self._format_prompt(ref_codes, ref_text, t) for t in batch]
            responses = self.backbone(prompts, gen_config=self.gen_config, do_preprocess=False)
            batch_wavs = [self._decode(r.text) for r in responses]
            if self.watermarker: batch_wavs = [self.watermarker.apply_watermark(w, self.sample_rate) for w in batch_wavs]
            all_wavs.extend(batch_wavs)
            if i + max_batch_size < len(texts) and torch.cuda.is_available(): torch.cuda.empty_cache()
        return all_wavs

    def infer_stream(self, text: str, ref_codes=None, ref_text=None, max_chars=256, voice=None, temperature=1.0, top_k=50, skip_normalize=False) -> Generator[np.ndarray, None, None]:
        if voice: ref_codes, ref_text = voice.get('codes', ref_codes), voice.get('text', ref_text)
        elif self._default_voice and (ref_codes is None or ref_text is None):
            v = self.get_preset_voice(None)
            ref_codes, ref_text = v['codes'], v['text']
        if ref_codes is None or ref_text is None: raise ValueError("Must provide voice info.")

        if not skip_normalize: text = self.normalizer.normalize(text)
        self.gen_config.temperature, self.gen_config.top_k = temperature, top_k
        chunks = split_text_into_chunks(text, max_chars=max_chars)
        for chunk in chunks:
            yield from self._infer_stream_single(chunk, ref_codes, ref_text)

    def _infer_stream_single(self, text, ref_codes, ref_text) -> Generator[np.ndarray, None, None]:
        if isinstance(ref_codes, (torch.Tensor, np.ndarray)): ref_codes = ref_codes.flatten().tolist()
        prompt = self._format_prompt(ref_codes, ref_text, text)
        audio_cache, token_cache = [], [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples, n_decoded_tokens = 0, len(ref_codes)

        for response in self.backbone.stream_infer([prompt], gen_config=self.gen_config, do_preprocess=False):
            new_tokens = response.text[len("".join(token_cache[len(ref_codes):])):] if len(token_cache) > len(ref_codes) else response.text
            if new_tokens: token_cache.append(new_tokens)

            if len(token_cache[n_decoded_tokens:]) >= self.streaming_frames_per_chunk + self.streaming_lookforward:
                ts = max(n_decoded_tokens - self.streaming_lookback - self.streaming_overlap_frames, 0)
                te = n_decoded_tokens + self.streaming_frames_per_chunk + self.streaming_lookforward + self.streaming_overlap_frames
                ss = (n_decoded_tokens - ts) * self.hop_length
                se = ss + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length

                recon = self._decode("".join(token_cache[ts:te]))
                audio_cache.append(recon[ss:se])
                processed = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
                new_end = len(audio_cache) * self.streaming_stride_samples
                yield processed[n_decoded_samples:new_end]
                n_decoded_samples, n_decoded_tokens = new_end, n_decoded_tokens + self.streaming_frames_per_chunk

        if len(token_cache) > n_decoded_tokens:
            rem = len(token_cache) - n_decoded_tokens
            ts = max(len(token_cache) - (self.streaming_lookback + self.streaming_overlap_frames + rem), 0)
            ss = (len(token_cache) - ts - rem - self.streaming_overlap_frames) * self.hop_length
            recon = self._decode("".join(token_cache[ts:]))
            audio_cache.append(recon[ss:])
            yield _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)[n_decoded_samples:]

    def cleanup_memory(self):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        import gc
        gc.collect()

    def get_optimization_stats(self) -> dict:
        return {
            'triton_enabled': self._triton_enabled,
            'max_batch_size': self.max_batch_size,
            'cached_references': len(self._ref_cache),
            'active_sessions': len(self.stored_dict),
            'kv_quant': getattr(self.gen_config, 'quant_policy', 0),
            'prefix_caching': True,
        }
