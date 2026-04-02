import os
import numpy as np
import logging
from typing import Optional, List, Any, Generator
from pathlib import Path
from .base import BaseVieneuTTS
from vieneu_utils.phonemize_text import phonemize_text
from vieneu_utils.core_utils import split_into_chunks_v2, get_silence_duration_v2

logger = logging.getLogger("Vieneu.Turbo")

class TurboGPUVieNeuTTS(BaseVieneuTTS):
    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS-v2-Turbo",
        decoder_repo: str = "pnnbao-ump/VieNeu-Codec",
        decoder_filename: str = "vieneu_decoder.onnx",
        encoder_repo: str = "pnnbao-ump/VieNeu-Codec",
        encoder_filename: str = "vieneu_encoder.onnx",
        device: str = "cuda",
        backend: str = "standard", # 'standard' (transformers) or 'lmdeploy'
        hf_token: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        # Normalize device: PyTorch expects 'cuda', Gradio might pass 'gpu'
        self.device = "cuda" if device.lower() in ["cuda", "gpu"] else "cpu"
        self.backend = backend.lower()
        self.backbone = None
        self.tokenizer = None
        self.decoder_sess = None
        self.encoder_sess = None
        self._is_onnx_codec = True
        
        # Load components
        self._load_backbone(backbone_repo, device, hf_token, **kwargs)
        self._load_decoder(decoder_repo, decoder_filename, device, hf_token)
        self._load_encoder(encoder_repo, encoder_filename, device, hf_token)
        
        # Load voices
        self._load_voices(backbone_repo, hf_token)

    def _load_backbone(self, repo, device, hf_token=None, **kwargs):
        if self.backend == "lmdeploy":
            try:
                from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
                logger.info(f"Loading Turbo GPU (LMDeploy) from: {repo}")
                
                engine_config = TurbomindEngineConfig(
                    cache_max_entry_count=kwargs.get("memory_util", 0.3),
                    tp=kwargs.get("tp", 1),
                    enable_prefix_caching=kwargs.get("enable_prefix_caching", True),
                    dtype='bfloat16',
                    quant_policy=kwargs.get("quant_policy", 0)
                )
                self.backbone = pipeline(repo, backend_config=engine_config)
                self.gen_config = GenerationConfig(
                    top_p=0.95, top_k=50, temperature=0.4, max_new_tokens=2048,
                    repetition_penalty=1.1,
                    do_sample=True, stop_words=["<|SPEECH_GENERATION_END|>"]
                )
            except ImportError:
                logger.warning("LMDeploy not found. Falling back to Standard (Transformers).")
                self.backend = "standard"

        if self.backend == "standard":
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading Turbo GPU (Standard) from: {repo} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(repo, token=hf_token)
            self.backbone = AutoModelForCausalLM.from_pretrained(
                repo, 
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                token=hf_token
            ).to(self.device)
            self.backbone.eval()

    def _load_decoder(self, decoder_repo, decoder_filename, device, hf_token=None):
        import onnxruntime as ort
        if os.path.exists(decoder_repo) and not os.path.isdir(decoder_repo):
            decoder_path = decoder_repo
        else:
            from huggingface_hub import hf_hub_download
            decoder_path = hf_hub_download(
                repo_id=decoder_repo, filename=decoder_filename, token=hf_token
            )
        
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device or "gpu" in device else ["CPUExecutionProvider"]
        self.decoder_sess = ort.InferenceSession(decoder_path, providers=providers)

    def _load_encoder(self, encoder_repo, encoder_filename, device, hf_token=None):
        import onnxruntime as ort
        if os.path.exists(encoder_repo) and not os.path.isdir(encoder_repo):
            encoder_path = encoder_repo
        else:
            from huggingface_hub import hf_hub_download
            try:
                encoder_path = hf_hub_download(
                    repo_id=encoder_repo, filename=encoder_filename, token=hf_token
                )
            except Exception:
                logger.warning("Speaker encoder not found for Turbo GPU.")
                return

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device or "gpu" in device else ["CPUExecutionProvider"]
        self.encoder_sess = ort.InferenceSession(encoder_path, providers=providers)

    def _get_voice_params(self, ref_codes: Any) -> np.ndarray:
        if isinstance(ref_codes, dict):
            ref_codes = ref_codes.get("codes")
        if isinstance(ref_codes, (np.ndarray, list)):
            emb = np.array(ref_codes, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb[None, :]
            return emb
        return np.zeros((1, 128), dtype=np.float32)

    def encode_reference(self, ref_audio: Any) -> np.ndarray:
        """Standalone ONNX implementation for speaker encoding in Turbo GPU mode."""
        if self.encoder_sess is None:
            raise RuntimeError("Speaker encoder model not loaded for Turbo GPU.")
        
        import librosa
        if isinstance(ref_audio, (str, Path)):
            wav, _ = librosa.load(ref_audio, sr=24000)
        else:
            wav = ref_audio
        
        if wav.ndim == 1:
            wav = wav[None, :]
        
        inputs = {"waveform": wav.astype(np.float32)}
        embedding = self.encoder_sess.run(None, inputs)[0]
        return embedding

    def _format_turbo_prompt(self, phonemes: str) -> str:
        return (
            f"<|speaker_16|>"
            f"<|TEXT_PROMPT_START|>{phonemes}<|TEXT_PROMPT_END|>"
            f"<|SPEECH_GENERATION_START|>"
        )

    def infer(self, text: str, voice: Optional[Any] = None, ref_codes: Optional[Any] = None, temperature: float = 0.4, top_k: int = 50, max_chars: int = 256, skip_normalize: bool = False, skip_phonemize: bool = False, **kwargs) -> np.ndarray:
        from vieneu_utils.phonemize_text import phonemize_text
        from vieneu_utils.core_utils import split_into_chunks_v2, get_silence_duration_v2

        phonemes = phonemize_text(text) if not skip_phonemize else text
        chunks = split_into_chunks_v2(phonemes, max_chunk_size=max_chars)
        
        if voice is None:
            voice = ref_codes
        if voice is None:
            voice = self.get_preset_voice()
        voice_embedding = self._get_voice_params(voice)

        all_wavs = []
        for i, chunk in enumerate(chunks):
            prompt = self._format_turbo_prompt(chunk.text)
            
            if self.backend == "lmdeploy":
                self.gen_config.temperature = temperature
                self.gen_config.top_k = top_k
                responses = self.backbone([prompt], gen_config=self.gen_config, do_preprocess=False)
                generated_text = responses[0].text
            else:
                import torch
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output_tokens = self.backbone.generate(
                        **inputs,
                        max_new_tokens=2048,
                        temperature=temperature,
                        top_k=top_k,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        # stop_strings=... is transformers >= 4.41.0
                        # For compatibility, we trim the output if eos is hit
                    )
                generated_text = self.tokenizer.decode(output_tokens[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            
            wav = self._decode(generated_text, voice_embedding)
            all_wavs.append(wav)

            if i < len(chunks) - 1:
                silence_dur = get_silence_duration_v2(chunk)
                if silence_dur > 0:
                    all_wavs.append(np.zeros(int(self.sample_rate * silence_dur), dtype=np.float32))

        final_wav = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]
        return self._apply_watermark(final_wav)

    def infer_batch(
        self, 
        texts: List[str], 
        voice: Optional[Any] = None, 
        ref_codes: Optional[Any] = None, 
        temperature: float = 0.4, 
        top_k: int = 50, 
        max_batch_size: int = 4, 
        skip_normalize: bool = False, 
        skip_phonemize: bool = False, 
        apply_watermark: bool = True, 
        **kwargs
    ) -> List[np.ndarray]:
        """Batch inference for Turbo GPU."""
        from vieneu_utils.phonemize_text import phonemize_batch
        
        if voice is None:
            voice = ref_codes
        if voice is None:
            voice = self.get_preset_voice()
        voice_embedding = self._get_voice_params(voice)

        chunk_phonemes = phonemize_batch(texts, skip_normalize=True) if not skip_phonemize else texts
        
        all_wavs = []
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i : i + max_batch_size]
            batch_ph = chunk_phonemes[i : i + max_batch_size]
            
            if self.backend == "lmdeploy":
                prompts = [self._format_turbo_prompt(ph) for ph in batch_ph]
                self.gen_config.temperature = temperature
                self.gen_config.top_k = top_k
                responses = self.backbone(prompts, gen_config=self.gen_config, do_preprocess=False)
                batch_codes = [r.text for r in responses]
                batch_wavs = [self._decode(c, voice_embedding) for c in batch_codes]
            else:
                # Standard sequential fallback for transformers
                batch_wavs = [
                    self.infer(
                        t, 
                        voice=voice, 
                        ref_codes=ref_codes, 
                        temperature=temperature, 
                        top_k=top_k, 
                        skip_normalize=True, 
                        skip_phonemize=True
                    ) for t in batch_texts
                ]
            
            if apply_watermark:
                batch_wavs = [self._apply_watermark(w) for w in batch_wavs]
            all_wavs.extend(batch_wavs)
        
        return all_wavs

    def _decode(self, codes_str: str, voice_embedding: np.ndarray) -> np.ndarray:
        from .utils import extract_speech_ids
        speech_ids = extract_speech_ids(codes_str)
        if not speech_ids:
            return np.array([], dtype=np.float32)
        
        tokens = np.array(speech_ids, dtype=np.int64)[None, :]
        inputs = {
            "content_ids": tokens, 
            "voice_embedding": voice_embedding
        }
        audio = self.decoder_sess.run(None, inputs)[0]
        return audio.flatten()

    def infer_stream(
        self,
        text: str,
        voice: Optional[Any] = None,
        ref_codes: Optional[Any] = None,
        temperature: float = 0.4,
        top_k: int = 50,
        max_chars: int = 256,
        skip_normalize: bool = False,
        skip_phonemize: bool = False,
        **kwargs
    ) -> Generator[np.ndarray, None, None]:
        """Streaming inference for Turbo GPU."""
        from vieneu_utils.phonemize_text import phonemize_text
        from vieneu_utils.core_utils import split_into_chunks_v2, get_silence_duration_v2

        phonemes = phonemize_text(text) if not skip_phonemize else text
        chunks = split_into_chunks_v2(phonemes, max_chunk_size=max_chars)
        
        if voice is None:
            voice = ref_codes
        if voice is None:
            voice = self.get_preset_voice()
        voice_embedding = self._get_voice_params(voice)

        for i, chunk in enumerate(chunks):
            prompt = self._format_turbo_prompt(chunk.text)
            
            if self.backend == "lmdeploy":
                # Ensure gen_config uses current parameters
                self.gen_config.temperature = temperature
                self.gen_config.top_k = top_k
                responses = self.backbone([prompt], gen_config=self.gen_config, do_preprocess=False)
                generated_text = responses[0].text
            else:
                import torch
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output_tokens = self.backbone.generate(
                        **inputs,
                        max_new_tokens=2048,
                        temperature=temperature,
                        top_k=top_k,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                generated_text = self.tokenizer.decode(output_tokens[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            
            wav = self._decode(generated_text, voice_embedding)
            yield self._apply_watermark(wav)

            if i < len(chunks) - 1:
                silence_dur = get_silence_duration_v2(chunk)
                if silence_dur > 0:
                    yield np.zeros(int(self.sample_rate * silence_dur), dtype=np.float32)

    def close(self):
        self.backbone = None
        self.decoder_sess = None
        self.encoder_sess = None



class TurboVieNeuTTS(BaseVieneuTTS):
    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF",
        backbone_filename: str = "vieneu-tts-v2-turbo.gguf",
        decoder_repo: str = "pnnbao-ump/VieNeu-Codec",
        decoder_filename: str = "vieneu_decoder.onnx",
        encoder_repo: str = "pnnbao-ump/VieNeu-Codec",
        encoder_filename: str = "vieneu_encoder.onnx",
        device: str = "cpu",
        hf_token: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.backbone = None
        self.decoder_sess = None
        self.encoder_sess = None
        self._is_onnx_codec = True
        self.max_context = 4096
        
        # Load components
        self._load_backbone(backbone_repo, backbone_filename, device, hf_token)
        self._load_decoder(decoder_repo, decoder_filename, device, hf_token)
        self._load_encoder(encoder_repo, encoder_filename, device, hf_token)
        
        # Load voices from the repository/directory (uses voices.json)
        self._load_voices(backbone_repo, hf_token)

    def _load_backbone(self, backbone_repo, backbone_filename, device, hf_token=None):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python is required for Turbo mode.")

        if os.path.exists(backbone_repo):
            model_path = backbone_repo
        else:
            from huggingface_hub import hf_hub_download
            try:
                model_path = hf_hub_download(
                    repo_id=backbone_repo, filename=backbone_filename, token=hf_token
                )
            except Exception:
                if os.path.exists(backbone_filename):
                    model_path = backbone_filename
                else:
                    raise FileNotFoundError(f"Neither repo '{backbone_repo}' nor '{backbone_filename}' found.")

        self.backbone = Llama(
            model_path=model_path,
            n_ctx=self.max_context,
            n_gpu_layers=-1 if device in ("gpu", "cuda") else 0,
            mlock=True,
            flash_attn=device in ("gpu", "cuda"),
            verbose=False,
        )

    def _load_decoder(self, decoder_repo, decoder_filename, device, hf_token=None):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for Turbo mode.")

        if os.path.exists(decoder_repo):
            decoder_path = decoder_repo
        else:
            from huggingface_hub import hf_hub_download
            try:
                decoder_path = hf_hub_download(
                    repo_id=decoder_repo, filename=decoder_filename, token=hf_token
                )
            except Exception:
                if os.path.exists(decoder_filename):
                    decoder_path = decoder_filename
                else:
                    raise FileNotFoundError(f"Neither repo '{decoder_repo}' nor '{decoder_filename}' found.")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device in ("gpu", "cuda") else ["CPUExecutionProvider"]
        self.decoder_sess = ort.InferenceSession(decoder_path, providers=providers)

    def _load_encoder(self, encoder_repo, encoder_filename, device, hf_token=None):
        try:
            import onnxruntime as ort
        except ImportError:
            return

        if os.path.exists(encoder_repo):
            encoder_path = encoder_repo
        else:
            from huggingface_hub import hf_hub_download
            try:
                encoder_path = hf_hub_download(
                    repo_id=encoder_repo, filename=encoder_filename, token=hf_token
                )
            except Exception:
                if os.path.exists(encoder_filename):
                    encoder_path = encoder_filename
                else:
                    logger.warning("Speaker encoder not found, voice cloning might be limited in Turbo mode.")
                    return

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device in ("gpu", "cuda") else ["CPUExecutionProvider"]
        self.encoder_sess = ort.InferenceSession(encoder_path, providers=providers)

    def encode_reference(self, ref_audio: Any) -> np.ndarray:
        """Standalone ONNX implementation for speaker encoding in Turbo mode."""
        if self.encoder_sess is None:
            raise RuntimeError("Speaker encoder model not loaded. Please ensure vieneu_encoder.onnx is available.")
        
        import librosa
        if isinstance(ref_audio, (str, Path)):
            wav, _ = librosa.load(ref_audio, sr=24000)
        else:
            wav = ref_audio
        
        if wav.ndim == 1:
            wav = wav[None, :]
        
        inputs = {"waveform": wav.astype(np.float32)}
        embedding = self.encoder_sess.run(None, inputs)[0]
        return embedding

    def _get_voice_params(self, ref_codes: Any) -> np.ndarray:
        """Extract the 128-dim voice embedding for the new decoder logic."""
        # Handle dict input (from get_preset_voice returned dict)
        if isinstance(ref_codes, dict):
            ref_codes = ref_codes.get("codes")
            
        # Ensure it is a float32 numpy array with shape (1, D)
        if isinstance(ref_codes, (np.ndarray, list)):
            emb = np.array(ref_codes, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb[None, :]
            if emb.shape[-1] in [128]:
                return emb
        
        # Fallback to zeros (128-dim)
        return np.zeros((1, 128), dtype=np.float32)

    def _decode(self, codes_str: str, voice_embedding: Optional[np.ndarray] = None) -> np.ndarray:
        from .utils import extract_speech_ids
        speech_ids = extract_speech_ids(codes_str)
        if not speech_ids:
            return np.array([], dtype=np.float32)
        
        tokens = np.array(speech_ids, dtype=np.int64)[None, :]
        
        if voice_embedding is None:
            voice_embedding = np.zeros((1, 128), dtype=np.float32)
            
        inputs = {
            "content_ids": tokens, 
            "voice_embedding": voice_embedding
        }
        audio = self.decoder_sess.run(None, inputs)[0]
        
        if audio.ndim == 3:
            return audio[0, 0, :]
        elif audio.ndim == 2:
            return audio[0, :]
        return audio.flatten()

    def infer(
        self,
        text: str,
        voice: Optional[Any] = None,
        ref_codes: Optional[Any] = None,
        temperature: float = 0.4,
        top_k: int = 50,
        max_chars: int = 256,
        skip_normalize: bool = False,
        skip_phonemize: bool = False,
        **kwargs
    ) -> np.ndarray:
        phonemes = phonemize_text(text) if not skip_phonemize else text

        chunks = split_into_chunks_v2(phonemes, max_chunk_size=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        # Use default voice if none provided
        if voice is None:
            voice = ref_codes
        if voice is None:
            voice = self.get_preset_voice()

        voice_embedding = self._get_voice_params(voice)

        all_wavs = []
        for i, chunk in enumerate(chunks):
            prompt = self._format_turbo_prompt(chunk.text)

            self.backbone.reset()
            result = self.backbone(
                prompt,
                max_tokens=2048,
                temperature=temperature,
                top_k=top_k,
                top_p=0.95,
                min_p=0.05,
                stop=["<|SPEECH_GENERATION_END|>"],
                repeat_penalty=1.15,
                echo=False,
            )
            wav = self._decode(result["choices"][0]["text"], voice_embedding)
            all_wavs.append(wav)

            if i < len(chunks) - 1:
                silence_dur = get_silence_duration_v2(chunk)
                if silence_dur > 0:
                    all_wavs.append(np.zeros(int(self.sample_rate * silence_dur), dtype=np.float32))

        final_wav = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]
        return self._apply_watermark(final_wav)

    def _format_turbo_prompt(self, phonemes: str) -> str:
        return (
            f"<|speaker_16|>"
            f"<|TEXT_PROMPT_START|>{phonemes}<|TEXT_PROMPT_END|>"
            f"<|SPEECH_GENERATION_START|>"
        )

    def infer_stream(
        self,
        text: str,
        voice: Optional[Any] = None,
        ref_codes: Optional[Any] = None,
        temperature: float = 0.4,
        top_k: int = 50,
        max_chars: int = 256,
        skip_normalize: bool = False,
        skip_phonemize: bool = False,
        **kwargs
    ) -> Generator[np.ndarray, None, None]:
        phonemes = phonemize_text(text) if not skip_phonemize else text

        chunks = split_into_chunks_v2(phonemes, max_chunk_size=max_chars)

        if voice is None:
            voice = ref_codes
        if voice is None:
            voice = self.get_preset_voice()

        voice_embedding = self._get_voice_params(voice)

        for i, chunk in enumerate(chunks):
            prompt = self._format_turbo_prompt(chunk.text)

            self.backbone.reset()
            result = self.backbone(
                prompt,
                max_tokens=2048,
                temperature=temperature,
                top_k=top_k,
                top_p=0.95,
                min_p=0.05,
                stop=["<|SPEECH_GENERATION_END|>"],
                repeat_penalty=1.15,
                echo=False,
            )
            wav = self._decode(result["choices"][0]["text"], voice_embedding)
            yield self._apply_watermark(wav)

            if i < len(chunks) - 1:
                silence_dur = get_silence_duration_v2(chunk)
                if silence_dur > 0:
                    yield np.zeros(int(self.sample_rate * silence_dur), dtype=np.float32)

    def infer_batch(self, texts: List[str], voice: Optional[Any] = None, ref_codes: Optional[Any] = None, apply_watermark: bool = True, **kwargs) -> List[np.ndarray]:
        results = [self.infer(t, voice=voice, ref_codes=ref_codes, **kwargs) for t in texts]
        if apply_watermark:
            results = [self._apply_watermark(r) for r in results]
        return results

    def close(self):
        if self.backbone:
            self.backbone.close()
            self.backbone = None
        self.decoder_sess = None
        self.encoder_sess = None
