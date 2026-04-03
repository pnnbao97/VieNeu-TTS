import os
import platform
from pathlib import Path
from typing import Optional, Union, List, Generator, Any, Dict
import numpy as np
import torch
import gc
import logging
from .base import BaseVieneuTTS
from .utils import extract_speech_ids, _linear_overlap_add, normalize_device
from vieneu_utils.phonemize_text import phonemize_with_dict, phonemize_batch
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks

logger = logging.getLogger("Vieneu.Standard")

class VieNeuTTS(BaseVieneuTTS):
    """
    Standard VieNeu-TTS implementation.
    Supports PyTorch + Transformers backend and GGUF quantized models.
    """

    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
        backbone_device: str = "cpu",
        codec_repo: str = "neuphonic/distill-neucodec",
        codec_device: str = "cpu",
        hf_token: Optional[str] = None,
    ):
        super().__init__()

        # Streaming configuration
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 25
        self.streaming_lookforward = 10
        self.streaming_lookback = 100
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        self._is_quantized_model = False
        self._is_onnx_codec = False
        self.tokenizer = None
        self.backbone = None
        self.codec = None

        if backbone_repo:
            self._load_backbone(backbone_repo, backbone_device, hf_token)
        self._load_codec(codec_repo, codec_device)
        self._load_voices(backbone_repo, hf_token)
        self._warmup_model()

    def _warmup_model(self) -> None:
        """Warm up the model to initialize CUDA/XPU kernels and KV cache."""
        try:
            logger.info("🔥 Warming up standard model...")
            dummy_text = "Xin chào"
            # Using very short dummy ref to speed up
            dummy_ref_codes = torch.zeros(10, dtype=torch.long)
            dummy_ref_text = "Chào"
            _ = self.infer(dummy_text, ref_codes=dummy_ref_codes, ref_text=dummy_ref_text, max_chars=16)
            logger.info("   ✅ Warmup complete")
        except Exception as e:
            logger.warning(f"   ⚠️ Warmup failed: {e}")

    def close(self) -> None:
        """Explicitly release model resources."""
        try:
            if self.backbone is not None:
                if self._is_quantized_model:
                    close_fn = getattr(self.backbone, "close", None)
                    if callable(close_fn):
                        close_fn()
                self.backbone = None

            if self.codec is not None:
                self.codec = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during VieNeuTTS closure: {e}")

    def _load_backbone(self, backbone_repo: str, backbone_device: str, hf_token: Optional[str] = None) -> None:
        backbone_device = normalize_device(backbone_device)
        logger.info(f"Loading backbone from: {backbone_repo} on {backbone_device} ...")

        if backbone_repo.lower().endswith("gguf") or "gguf" in backbone_repo.lower():
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "Failed to import `llama_cpp`. Please install llama-cpp-python version >= 0.3.16."
                ) from e
            self.backbone = Llama.from_pretrained(
                repo_id=backbone_repo,
                filename="*.gguf",
                verbose=False,
                n_gpu_layers=-1 if backbone_device in ("gpu", "cuda") else 0,
                n_ctx=self.max_context,
                mlock=True,
                flash_attn=True if backbone_device in ("gpu", "cuda") else False,
                token=hf_token,
            )
            self._is_quantized_model = True
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo, token=hf_token)

            # Configure tokenizer for batching
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.backbone = AutoModelForCausalLM.from_pretrained(backbone_repo, token=hf_token).to(
                torch.device(backbone_device)
            )

            # Optional torch.compile for non-Windows/non-Mac platforms if desired
            if os.getenv("VIENEU_COMPILE") == "1" and platform.system() == "Linux":
                try:
                    logger.info("🚀 Compiling backbone model with torch.compile...")
                    self.backbone = torch.compile(self.backbone, mode="reduce-overhead")
                except Exception as e:
                    logger.warning(f"Failed to compile backbone: {e}")

    def _load_codec(self, codec_repo: str, codec_device: str) -> None:
        super()._load_codec(codec_repo, codec_device)

    def load_lora_adapter(self, lora_repo_id: str, hf_token: Optional[str] = None) -> bool:
        if self._is_quantized_model:
            raise NotImplementedError("LoRA not supported for GGUF quantized models. Use PyTorch backbone.")

        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError("PEFT library required for LoRA. Install with: pip install peft")

        logger.info(f"🎯 Loading LoRA adapter from: {lora_repo_id}")

        if not hasattr(self, '_lora_loaded') or not self._lora_loaded:
            self._current_lora_repo = None
            self._lora_loaded = False

        if self._lora_loaded:
            self.unload_lora_adapter()

        try:
            self.backbone = PeftModel.from_pretrained(self.backbone, lora_repo_id, token=hf_token)
            self._lora_loaded = True
            self._current_lora_repo = lora_repo_id
            self._load_voices(lora_repo_id, hf_token, clear_existing=True)
            logger.info(f"   ✅ LoRA adapter loaded: {lora_repo_id}")
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA adapter: {str(e)}") from e

    def unload_lora_adapter(self) -> bool:
        if not getattr(self, '_lora_loaded', False):
            return False

        logger.info(f"   🔄 Unloading LoRA adapter: {self._current_lora_repo}")
        try:
            self.backbone = self.backbone.unload()
            self._lora_loaded = False
            self._current_lora_repo = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("   ✅ LoRA adapter unloaded, original weights restored")
            return True
        except Exception as e:
            logger.error(f"   ⚠️ Error during unload: {e}")
            return False

    def infer(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, max_chars: int = 256, silence_p: float = 0.15, crossfade_p: float = 0.0, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False, apply_watermark: bool = True) -> np.ndarray:

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        if len(chunks) == 1:
            ref_phonemes = self.get_ref_phonemes(ref_text)
            phonemes = phonemize_with_dict(chunks[0], skip_normalize=True)
            if self._is_quantized_model:
                output_str = self._infer_ggml(ref_codes, ref_phonemes, phonemes, temperature, top_k)
            else:
                prompt_ids = self._apply_chat_template(ref_codes, ref_phonemes, phonemes)
                output_str = self._infer_torch(prompt_ids, temperature, top_k)
            wav = self._decode(output_str)
            if apply_watermark:
                wav = self._apply_watermark(wav)
            return wav

        all_wavs = self.infer_batch(
            chunks,
            ref_codes=ref_codes,
            ref_text=ref_text,
            temperature=temperature,
            top_k=top_k,
            skip_normalize=True,
            apply_watermark=False
        )
        final_wav = join_audio_chunks(all_wavs, self.sample_rate, silence_p, crossfade_p)
        if apply_watermark:
            final_wav = self._apply_watermark(final_wav)
        return final_wav

    def infer_batch(self, texts: List[str], ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False, apply_watermark: bool = True) -> List[np.ndarray]:
        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            texts = [self.normalizer.normalize(t) for t in texts]

        ref_phonemes = self.get_ref_phonemes(ref_text)
        chunk_phonemes = phonemize_batch(texts, skip_normalize=True)

        all_wavs = []
        # If model is GGUF, we still process sequentially for now as llama-cpp-python batching for TTS is complex
        if self._is_quantized_model:
            for phonemes in chunk_phonemes:
                output_str = self._infer_ggml(ref_codes, ref_phonemes, phonemes, temperature, top_k)
                wav = self._decode(output_str)
                if apply_watermark:
                    wav = self._apply_watermark(wav)
                all_wavs.append(wav)
        # If model is Torch, we can leverage true batch generation
        else:
            batch_prompt_ids = []
            for phonemes in chunk_phonemes:
                prompt_ids = self._apply_chat_template(ref_codes, ref_phonemes, phonemes)
                batch_prompt_ids.append(torch.tensor(prompt_ids))

            inputs = self.tokenizer.pad(
                {"input_ids": batch_prompt_ids},
                padding=True,
                return_tensors="pt"
            )
            # Move all tensors to device
            inputs = {k: v.to(self.backbone.device) for k, v in inputs.items()}

            speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
            with torch.no_grad():
                output_tokens = self.backbone.generate(
                    **inputs,
                    max_length=self.max_context,
                    eos_token_id=speech_end_id,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    use_cache=True,
                    min_new_tokens=50,
                )

            input_length = inputs["input_ids"].shape[-1]
            for i in range(len(texts)):
                generated_ids = output_tokens[i, input_length:]
                output_str = self.tokenizer.decode(generated_ids, add_special_tokens=False)
                wav = self._decode(output_str)
                if apply_watermark:
                    wav = self._apply_watermark(wav)
                all_wavs.append(wav)

        return all_wavs

    def infer_stream(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, max_chars: int = 256, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False) -> Generator[np.ndarray, None, None]:

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks:
            return

        # Pre-phonemize all inputs for performance
        ref_phonemes = self.get_ref_phonemes(ref_text)
        chunk_phonemes = phonemize_batch(chunks, skip_normalize=True)

        for phonemes in chunk_phonemes:
            if self._is_quantized_model:
                yield from self._infer_stream_ggml(ref_codes, ref_phonemes, phonemes, temperature, top_k)
            else:
                prompt_ids = self._apply_chat_template(ref_codes, ref_phonemes, phonemes)
                output_str = self._infer_torch(prompt_ids, temperature, top_k)
                wav = self._decode(output_str)
                yield self._apply_watermark(wav)

    def _apply_chat_template(self, ref_codes: Any, ref_phonemes: str, chunk_phonemes: str) -> List[int]:
        ref_codes_list = self.to_list(ref_codes)
        full_phonemes = f"{ref_phonemes} {chunk_phonemes}"

        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

        input_ids = self.tokenizer.encode(full_phonemes, add_special_tokens=False)
        chat = "user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"
        ids = self.tokenizer.encode(chat)

        text_replace_idx = ids.index(text_replace)
        ids = ids[:text_replace_idx] + [text_prompt_start] + input_ids + [text_prompt_end] + ids[text_replace_idx + 1:]

        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes_list])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)
        return ids

    def _infer_torch(self, prompt_ids: List[int], temperature: float = 1.0, top_k: int = 50) -> str:
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                prompt_tensor,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                use_cache=True,
                min_new_tokens=50,
            )
        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(output_tokens[0, input_length:].cpu().numpy().tolist(), add_special_tokens=False)
        return output_str

    def _infer_ggml(self, ref_codes: Any, ref_phonemes: str, chunk_phonemes: str, temperature: float = 1.0, top_k: int = 50) -> str:
        ref_codes_list = self.to_list(ref_codes)
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes_list])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_phonemes} {chunk_phonemes}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )
        output = self.backbone(prompt, max_tokens=self.max_context, temperature=temperature, top_k=top_k, stop=["<|SPEECH_GENERATION_END|>"])
        return output["choices"][0]["text"]

    def _infer_stream_ggml(self, ref_codes: Any, ref_phonemes: str, chunk_phonemes: str, temperature: float = 1.0, top_k: int = 50) -> Generator[np.ndarray, None, None]:

        ref_codes_list = self.to_list(ref_codes)
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes_list])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_phonemes} {chunk_phonemes}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        audio_cache: List[np.ndarray] = []
        token_cache: List[str] = [f"<|speech_{idx}|>" for idx in ref_codes_list]
        n_decoded_samples: int = 0
        n_decoded_tokens: int = len(ref_codes_list)

        for item in self.backbone(prompt, max_tokens=self.max_context, temperature=temperature, top_k=top_k, stop=["<|SPEECH_GENERATION_END|>"], stream=True):
            output_str = item["choices"][0]["text"]
            token_cache.append(output_str)

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
