from pathlib import Path
from typing import Optional, Union, List, Generator, Any, Dict
import numpy as np
import requests
import json
import asyncio
import logging
from .base import BaseVieneuTTS
from .utils import _linear_overlap_add
from vieneu_utils.phonemize_text import phonemize_with_dict, phonemize_batch
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks

logger = logging.getLogger("Vieneu.Remote")

class RemoteVieNeuTTS(BaseVieneuTTS):
    """
    Client for VieNeu-TTS running on a remote LMDeploy server.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:23333/v1",
        model_name: str = "pnnbao-ump/VieNeu-TTS",
        codec_repo: str = "neuphonic/distill-neucodec",
        codec_device: str = "cpu",
        hf_token: Optional[str] = None
    ):
        self.api_base = api_base.rstrip('/')
        self.model_name = model_name

        super().__init__(
            codec_repo=codec_repo,
            codec_device=codec_device
        )
        # Override some streaming defaults for remote
        self.streaming_frames_per_chunk = 10
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        self._load_voices_from_repo(model_name, hf_token)

    def _load_backbone(self, backbone_repo, backbone_device, hf_token=None):
        pass


    def infer(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, 'torch.Tensor']] = None, ref_text: Optional[str] = None, max_chars: int = 256, silence_p: float = 0.15, crossfade_p: float = 0.0, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False, apply_watermark: bool = True, **kwargs) -> np.ndarray:

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        if len(chunks) == 1:
            prompt = self._format_prompt(ref_codes, ref_text, chunks[0])
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": temperature,
                "top_k": top_k,
                "stop": ["<|SPEECH_GENERATION_END|>"],
                "stream": False
            }
            try:
                response = requests.post(f"{self.api_base}/chat/completions", json=payload, timeout=60)
                response.raise_for_status()
                output_str = response.json()["choices"][0]["message"]["content"]
                wav = self._decode(output_str)
                if apply_watermark:
                    wav = self._apply_watermark(wav)
                return wav
            except Exception as e:
                logger.error(f"Error during remote inference: {e}")
                return np.array([], dtype=np.float32)

        # For multiple chunks, use async for parallel processing
        return asyncio.run(self.infer_async(
            text, ref_codes=ref_codes, ref_text=ref_text,
            max_chars=max_chars, silence_p=silence_p, crossfade_p=crossfade_p,
            temperature=temperature, top_k=top_k, skip_normalize=True, apply_watermark=True
        ))

    def infer_stream(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, 'torch.Tensor']] = None, ref_text: Optional[str] = None, max_chars: int = 256, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False) -> Generator[np.ndarray, None, None]:

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        for chunk in chunks:
            yield from self._infer_stream_chunk(chunk, ref_codes, ref_text, temperature, top_k)

    def _infer_stream_chunk(self, chunk, ref_codes, ref_text, temperature, top_k):
        prompt = self._format_prompt(ref_codes, ref_text, chunk)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": temperature,
            "top_k": top_k,
            "stop": ["<|SPEECH_GENERATION_END|>"],
            "stream": True
        }

        ref_codes_list = self.to_list(ref_codes)

        audio_cache: List[np.ndarray] = []
        token_cache: List[str] = [f"<|speech_{idx}|>" for idx in ref_codes_list]
        n_decoded_samples: int = 0
        n_decoded_tokens: int = len(ref_codes_list)

        try:
             with requests.post(f"{self.api_base}/chat/completions", json=payload, stream=True, timeout=60) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line: continue
                    line_str = line.decode('utf-8')
                    if not line_str.startswith('data: '): continue
                    data_str = line_str[6:]
                    if data_str == '[DONE]': break
                    try:
                        content = json.loads(data_str)["choices"][0]["delta"].get("content", "")
                        if content:
                             token_cache.append(content)
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
                    except json.JSONDecodeError: continue
        except Exception as e:
            logger.error(f"Error streaming chunk: {e}")
            return

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

    async def infer_async(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, 'torch.Tensor']] = None, ref_text: Optional[str] = None, max_chars: int = 256, silence_p: float = 0.15, crossfade_p: float = 0.0, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, session=None, skip_normalize: bool = False, apply_watermark: bool = True) -> np.ndarray:
        try:
            import aiohttp
        except ImportError:
            raise ImportError("Async requires 'aiohttp'.")

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        should_close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            should_close_session = True

        try:
            tasks = [self._infer_chunk_async(session, chunk, ref_codes, ref_text, temperature, top_k) for chunk in chunks]
            wavs = await asyncio.gather(*tasks)
            final_wav = join_audio_chunks(wavs, self.sample_rate, silence_p, crossfade_p)
            if apply_watermark:
                final_wav = self._apply_watermark(final_wav)
            return final_wav
        finally:
            if should_close_session:
                await session.close()

    def infer_batch(self, texts: List[str], ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, 'torch.Tensor']] = None, ref_text: Optional[str] = None, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False, apply_watermark: bool = True) -> List[np.ndarray]:
        """Synchronous wrapper for async batch inference."""
        return asyncio.run(self.infer_batch_async(
            texts, ref_audio=ref_audio, ref_codes=ref_codes, ref_text=ref_text,
            voice=voice, temperature=temperature, top_k=top_k, skip_normalize=skip_normalize,
            apply_watermark=apply_watermark
        ))

    async def _infer_chunk_async(
        self,
        session,
        chunk: str,
        ref_codes: Union[List[int], 'torch.Tensor', np.ndarray],
        ref_text: str,
        temperature: float,
        top_k: int,
        ref_phonemes: Optional[str] = None,
        chunk_phonemes: Optional[str] = None
    ) -> np.ndarray:
        """Internal helper for asynchronous chunk inference."""
        prompt = self._format_prompt(ref_codes, ref_text, chunk, ref_phonemes=ref_phonemes, input_phonemes=chunk_phonemes)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": temperature,
            "top_k": top_k,
            "stop": ["<|SPEECH_GENERATION_END|>"],
            "stream": False
        }
        try:
            async with session.post(f"{self.api_base}/chat/completions", json=payload, timeout=60) as resp:
                resp.raise_for_status()
                data = await resp.json()
                output_str = data["choices"][0]["message"]["content"]
                return self._decode(output_str)
        except Exception as e:
            logger.error(f"Error in async chunk: {e}")
            return np.array([], dtype=np.float32)

    async def infer_batch_async(self, texts: List[str], ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, 'torch.Tensor']] = None, ref_text: Optional[str] = None, max_chars: int = 256, silence_p: float = 0.15, crossfade_p: float = 0.0, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, concurrency_limit: int = 50, skip_normalize: bool = False, apply_watermark: bool = True) -> List[np.ndarray]:
        try:
            import aiohttp
        except ImportError:
            raise ImportError("Async requires 'aiohttp'.")

        if not skip_normalize:
            texts = [self.normalizer.normalize(t) for t in texts]

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        # Pre-phonemize all for performance
        ref_phonemes = self.get_ref_phonemes(ref_text)
        all_phonemes = phonemize_batch(texts, skip_normalize=True)

        sem = asyncio.Semaphore(concurrency_limit)
        async with aiohttp.ClientSession() as session:
            async def bounded_infer(text, ph):
                async with sem:
                    # Split into chunks internally if needed
                    chunks = split_text_into_chunks(text, max_chars=max_chars)
                    if not chunks: return np.array([], dtype=np.float32)

                    if len(chunks) == 1:
                        wav = await self._infer_chunk_async(session, chunks[0], ref_codes, ref_text, temperature, top_k, ref_phonemes=ref_phonemes, chunk_phonemes=ph)
                        if apply_watermark: wav = self._apply_watermark(wav)
                        return wav

                    # Re-phonemize chunks if splitting happened
                    chunk_phonemes = phonemize_batch(chunks, skip_normalize=True)
                    tasks = [self._infer_chunk_async(session, c, ref_codes, ref_text, temperature, top_k, ref_phonemes=ref_phonemes, chunk_phonemes=c_ph)
                            for c, c_ph in zip(chunks, chunk_phonemes)]
                    wavs = await asyncio.gather(*tasks)
                    final_wav = join_audio_chunks(wavs, self.sample_rate, silence_p, crossfade_p)
                    if apply_watermark: final_wav = self._apply_watermark(final_wav)
                    return final_wav

            tasks = [bounded_infer(text, ph) for text, ph in zip(texts, all_phonemes)]
            results = await asyncio.gather(*tasks)

        return results
