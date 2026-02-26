from typing import Generator, Union, Optional
from pathlib import Path
import re
import numpy as np
import torch
from neucodec import NeuCodec, DistillNeuCodec
from vieneu_utils.phonemize_text import phonemize_with_dict
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks
from .base import VieneuBase, _linear_overlap_add, SPEECH_TOKEN_RE

class VieNeuTTS(VieneuBase):
    """
    Standard VieNeu-TTS implementation.
    Supports PyTorch + Transformers and GGUF quantized models.
    """

    def __init__(
        self,
        backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
        backbone_device="cpu",
        codec_repo="neuphonic/distill-neucodec",
        codec_device="cpu",
        hf_token=None,
    ):
        super().__init__()

        # Streaming settings
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 25
        self.streaming_lookforward = 10
        self.streaming_lookback = 100
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        self._is_quantized_model = False
        self._is_onnx_codec = False

        if backbone_repo:
            self._load_backbone(backbone_repo, backbone_device, hf_token)
        self._load_codec(codec_repo, codec_device)
        self._load_voices(backbone_repo, hf_token)

    def _load_backbone(self, backbone_repo, backbone_device, hf_token=None):
        if backbone_device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available, falling back to CPU")
            backbone_device = "cpu"

        print(f"Loading backbone from: {backbone_repo} on {backbone_device} ...")

        if backbone_repo.lower().endswith("gguf") or "gguf" in backbone_repo.lower():
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "Failed to import `llama_cpp`. "
                    "Install llama-cpp-python >= 0.3.16."
                ) from e
            self.backbone = Llama.from_pretrained(
                repo_id=backbone_repo,
                filename="*.gguf",
                verbose=False,
                n_gpu_layers=-1 if backbone_device == "gpu" else 0,
                n_ctx=self.max_context,
                mlock=True,
                flash_attn=True if backbone_device == "gpu" else False,
                token=hf_token,
            )
            self._is_quantized_model = True
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo, token=hf_token)
            self.backbone = AutoModelForCausalLM.from_pretrained(backbone_repo, token=hf_token).to(
                torch.device(backbone_device)
            )

    def _load_codec(self, codec_repo, codec_device):
        if codec_device == "mps" and not torch.backends.mps.is_available():
            codec_device = "cpu"

        print(f"Loading codec from: {codec_repo} on {codec_device} ...")
        match codec_repo:
            case "neuphonic/neucodec":
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/distill-neucodec":
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/neucodec-onnx-decoder-int8":
                if codec_device != "cpu":
                    raise ValueError("Onnx decoder only currently runs on CPU.")
                from neucodec import NeuCodecOnnxDecoder
                self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
                self._is_onnx_codec = True
            case _:
                raise ValueError(f"Unsupported codec repository: {codec_repo}")

    def load_lora_adapter(self, lora_repo_id: str, hf_token: str = None):
        if self._is_quantized_model:
            raise NotImplementedError("LoRA not supported for GGUF models. Use PyTorch backbone.")
        from peft import PeftModel
        print(f"🎯 Loading LoRA adapter from: {lora_repo_id}")
        if self.backbone.__class__.__name__ == "PeftModel":
            self.unload_lora_adapter()
        self.backbone = PeftModel.from_pretrained(self.backbone, lora_repo_id, token=hf_token)
        self._load_voices(lora_repo_id, hf_token, clear_existing=True)
        return True

    def unload_lora_adapter(self):
        if self.backbone.__class__.__name__ != "PeftModel":
            return False
        print(f"   🔄 Unloading LoRA adapter")
        self.backbone = self.backbone.unload()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def _decode(self, codes: str):
        speech_ids = [int(num) for num in SPEECH_TOKEN_RE.findall(codes)]
        if not speech_ids:
            raise ValueError("No valid speech tokens found in the output.")

        if self._is_onnx_codec:
            codes_np = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            recon = self.codec.decode_code(codes_np)
        else:
            with torch.no_grad():
                codes_torch = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(self.codec.device)
                recon = self.codec.decode_code(codes_torch).cpu().numpy()
        return recon[0, 0, :]

    def _apply_chat_template(self, ref_codes: Union[list, torch.Tensor, np.ndarray], ref_text: str, input_text: str) -> list:
        if isinstance(ref_codes, (torch.Tensor, np.ndarray)):
            ref_codes = ref_codes.flatten().tolist()

        full_text = phonemize_with_dict(ref_text) + " " + phonemize_with_dict(input_text, skip_normalize=True)
        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

        chat = "user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"
        ids = self.tokenizer.encode(chat)

        tr_idx = ids.index(text_replace)
        ids = ids[:tr_idx] + [text_prompt_start] + input_ids + [text_prompt_end] + ids[tr_idx+1:]

        sr_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes_ids = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:sr_idx] + [speech_gen_start] + codes_ids
        return ids

    def _infer_torch(self, prompt_ids: list, temperature: float = 1.0, top_k: int = 50) -> str:
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
        return self.tokenizer.decode(output_tokens[0, input_length:].cpu().numpy().tolist(), add_special_tokens=False)

    def _infer_ggml(self, ref_codes, ref_text, input_text, temperature=1.0, top_k=50) -> str:
        if isinstance(ref_codes, (torch.Tensor, np.ndarray)):
            ref_codes = ref_codes.flatten().tolist()
        ref_text = phonemize_with_dict(ref_text)
        input_text = phonemize_with_dict(input_text, skip_normalize=True)
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
                  f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}")
        output = self.backbone(prompt, max_tokens=self.max_context, temperature=temperature, top_k=top_k, stop=["<|SPEECH_GENERATION_END|>"])
        return output["choices"][0]["text"]

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

        if not skip_normalize:
            text = self.normalizer.normalize(text)
        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks: return np.array([], dtype=np.float32)

        all_wavs = []
        for chunk in chunks:
            output_str = self._infer_ggml(ref_codes, ref_text, chunk, temperature, top_k) if self._is_quantized_model else \
                         self._infer_torch(self._apply_chat_template(ref_codes, ref_text, chunk), temperature, top_k)
            wav = self._decode(output_str)
            all_wavs.append(wav)

        final_wav = join_audio_chunks(all_wavs, self.sample_rate, silence_p, crossfade_p)
        if self.watermarker:
            final_wav = self.watermarker.apply_watermark(final_wav, sample_rate=self.sample_rate)
        return final_wav

    def infer_stream(self, text: str, ref_codes=None, ref_text=None, max_chars=256, voice=None, temperature=1.0, top_k=50, skip_normalize=False) -> Generator[np.ndarray, None, None]:
        if voice:
            ref_codes, ref_text = voice.get('codes', ref_codes), voice.get('text', ref_text)
        elif self._default_voice and (ref_codes is None or ref_text is None):
            v = self.get_preset_voice(None)
            ref_codes, ref_text = v['codes'], v['text']
        if ref_codes is None or ref_text is None:
             raise ValueError("Must provide either 'voice' dict or both 'ref_codes' and 'ref_text'.")

        if not skip_normalize:
            text = self.normalizer.normalize(text)
        chunks = split_text_into_chunks(text, max_chars=max_chars)

        for chunk in chunks:
            if self._is_quantized_model:
                yield from self._infer_stream_ggml(ref_codes, ref_text, chunk, temperature, top_k)
            else:
                wav = self._decode(self._infer_torch(self._apply_chat_template(ref_codes, ref_text, chunk), temperature, top_k))
                if self.watermarker: wav = self.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
                yield wav

    def _infer_stream_ggml(self, ref_codes, ref_text, input_text, temperature=1.0, top_k=50) -> Generator[np.ndarray, None, None]:
        if isinstance(ref_codes, (torch.Tensor, np.ndarray)):
            ref_codes = ref_codes.flatten().tolist()
        ref_text_p = phonemize_with_dict(ref_text)
        input_text_p = phonemize_with_dict(input_text, skip_normalize=True)
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text_p} {input_text_p}"
                  f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}")

        audio_cache, token_cache = [], [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples, n_decoded_tokens = 0, len(ref_codes)

        for item in self.backbone(prompt, max_tokens=self.max_context, temperature=temperature, top_k=top_k, stop=["<|SPEECH_GENERATION_END|>"], stream=True):
            token_cache.append(item["choices"][0]["text"])
            if len(token_cache[n_decoded_tokens:]) >= self.streaming_frames_per_chunk + self.streaming_lookforward:
                ts = max(n_decoded_tokens - self.streaming_lookback - self.streaming_overlap_frames, 0)
                te = n_decoded_tokens + self.streaming_frames_per_chunk + self.streaming_lookforward + self.streaming_overlap_frames
                ss = (n_decoded_tokens - ts) * self.hop_length
                se = ss + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length

                recon = self._decode("".join(token_cache[ts:te]))
                if self.watermarker: recon = self.watermarker.apply_watermark(recon, sample_rate=self.sample_rate)
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
            if self.watermarker: recon = self.watermarker.apply_watermark(recon, sample_rate=self.sample_rate)
            audio_cache.append(recon[ss:])
            yield _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)[n_decoded_samples:]
