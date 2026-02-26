from typing import Generator, Union, Optional
from pathlib import Path
import json
import requests
import asyncio
import numpy as np
import torch
from .standard import VieNeuTTS
from .base import _linear_overlap_add, SPEECH_TOKEN_RE
from vieneu_utils.phonemize_text import phonemize_with_dict
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks

class RemoteVieNeuTTS(VieNeuTTS):
    """
    Client for VieNeu-TTS running on a remote LMDeploy server.
    """

    def __init__(
        self,
        api_base="http://localhost:23333/v1",
        model_name="pnnbao-ump/VieNeu-TTS",
        codec_repo="neuphonic/distill-neucodec",
        codec_device="cpu",
        hf_token=None
    ):
        self.api_base = api_base.rstrip('/')
        self.model_name = model_name

        # Initialize without backbone
        super().__init__(
            backbone_repo=None,
            codec_repo=codec_repo,
            codec_device=codec_device,
            hf_token=hf_token
        )

        self.streaming_frames_per_chunk = 10
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        self._load_voices_from_repo(model_name, hf_token)
        print(f"📡 RemoteVieNeuTTS ready! Using backend: {self.api_base}")

    def _load_backbone(self, backbone_repo, backbone_device, hf_token=None):
        pass

    def _format_prompt(self, ref_codes: list, ref_text: str, input_text: str) -> str:
        rtp, itp = phonemize_with_dict(ref_text), phonemize_with_dict(input_text, skip_normalize=True)
        cs = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        return (f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{rtp} {itp}"
                f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{cs}")

    def infer(self, text: str, ref_audio=None, ref_codes=None, ref_text=None, max_chars=256, silence_p=0.15, crossfade_p=0.0, voice=None, temperature=1.0, top_k=50, skip_normalize=False) -> np.ndarray:
        if voice: ref_codes, ref_text = voice.get('codes', ref_codes), voice.get('text', ref_text)
        if ref_audio is not None and ref_codes is None: ref_codes = self.encode_reference(ref_audio)
        elif self._default_voice and (ref_codes is None or ref_text is None):
            v = self.get_preset_voice(None); ref_codes, ref_text = v['codes'], v['text']
        if ref_codes is None or ref_text is None: raise ValueError("Must provide voice info.")

        if not skip_normalize: text = self.normalizer.normalize(text)
        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks: return np.array([], dtype=np.float32)

        all_wavs = []
        for chunk in chunks:
            if isinstance(ref_codes, (torch.Tensor, np.ndarray)): ref_codes_list = ref_codes.flatten().tolist()
            else: ref_codes_list = ref_codes
            prompt = self._format_prompt(ref_codes_list, ref_text, chunk)
            payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048, "temperature": temperature, "top_k": top_k, "stop": ["<|SPEECH_GENERATION_END|>"], "stream": False}
            try:
                response = requests.post(f"{self.api_base}/chat/completions", json=payload, timeout=60)
                response.raise_for_status()
                all_wavs.append(self._decode(response.json()["choices"][0]["message"]["content"]))
            except Exception as e:
                print(f"Error during remote inference: {e}")

        final_wav = join_audio_chunks(all_wavs, self.sample_rate, silence_p, crossfade_p)
        if self.watermarker: final_wav = self.watermarker.apply_watermark(final_wav, sample_rate=self.sample_rate)
        return final_wav

    def infer_stream(self, text: str, ref_audio=None, ref_codes=None, ref_text=None, max_chars=256, voice=None, temperature=1.0, top_k=50, skip_normalize=False) -> Generator[np.ndarray, None, None]:
        if voice: ref_codes, ref_text = voice.get('codes', ref_codes), voice.get('text', ref_text)
        if ref_audio is not None and ref_codes is None: ref_codes = self.encode_reference(ref_audio)
        elif self._default_voice and (ref_codes is None or ref_text is None):
            v = self.get_preset_voice(None); ref_codes, ref_text = v['codes'], v['text']
        if ref_codes is None or ref_text is None: raise ValueError("Must provide voice info.")
        if not skip_normalize: text = self.normalizer.normalize(text)
        chunks = split_text_into_chunks(text, max_chars=max_chars)
        for chunk in chunks:
            yield from self._infer_stream_chunk(chunk, ref_codes, ref_text, temperature, top_k)

    def _infer_stream_chunk(self, chunk, ref_codes, ref_text, temperature, top_k):
        if isinstance(ref_codes, (torch.Tensor, np.ndarray)): rcl = ref_codes.flatten().tolist()
        else: rcl = ref_codes
        prompt = self._format_prompt(rcl, ref_text, chunk)
        payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048, "temperature": temperature, "top_k": top_k, "stop": ["<|SPEECH_GENERATION_END|>"], "stream": True}
        audio_cache, token_cache = [], [f"<|speech_{idx}|>" for idx in rcl]
        n_decoded_samples, n_decoded_tokens = 0, len(rcl)
        try:
             with requests.post(f"{self.api_base}/chat/completions", json=payload, stream=True, timeout=60) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line: continue
                    ls = line.decode('utf-8')
                    if not ls.startswith('data: '): continue
                    ds = ls[6:]
                    if ds == '[DONE]': break
                    try:
                        content = json.loads(ds)["choices"][0]["delta"].get("content", "")
                        if content:
                             token_cache.append(content)
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
                    except json.JSONDecodeError: continue
        except Exception as e: print(f"Error streaming chunk: {e}"); return
        if (rem := len(token_cache) - n_decoded_tokens) > 0:
            ts = max(len(token_cache) - (self.streaming_lookback + self.streaming_overlap_frames + rem), 0)
            ss = (len(token_cache) - ts - rem - self.streaming_overlap_frames) * self.hop_length
            recon = self._decode("".join(token_cache[ts:]))
            if self.watermarker: recon = self.watermarker.apply_watermark(recon, sample_rate=self.sample_rate)
            audio_cache.append(recon[ss:])
            yield _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)[n_decoded_samples:]

    async def infer_async(self, text: str, ref_audio=None, ref_codes=None, ref_text=None, max_chars=256, silence_p=0.15, crossfade_p=0.0, voice=None, temperature=1.0, top_k=50, session=None, skip_normalize=False) -> np.ndarray:
        import aiohttp
        if voice: ref_codes, ref_text = voice.get('codes', ref_codes), voice.get('text', ref_text)
        if ref_audio is not None and ref_codes is None: ref_codes = self.encode_reference(ref_audio)
        elif self._default_voice and (ref_codes is None or ref_text is None):
            v = self.get_preset_voice(None); ref_codes, ref_text = v['codes'], v['text']
        if ref_codes is None or ref_text is None: raise ValueError("Must provide voice info.")
        if not skip_normalize: text = self.normalizer.normalize(text)
        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks: return np.array([], dtype=np.float32)
        sc = False
        if session is None: session = aiohttp.ClientSession(); sc = True
        try:
            tasks = [self._infer_chunk_async(session, c, ref_codes, ref_text, temperature, top_k) for c in chunks]
            wavs = await asyncio.gather(*tasks)
            final_wav = join_audio_chunks(wavs, self.sample_rate, silence_p, crossfade_p)
            if self.watermarker: final_wav = self.watermarker.apply_watermark(final_wav, sample_rate=self.sample_rate)
            return final_wav
        finally:
            if sc: await session.close()

    async def _infer_chunk_async(self, session, chunk, ref_codes, ref_text, temperature, top_k):
        if isinstance(ref_codes, (torch.Tensor, np.ndarray)): rcl = ref_codes.flatten().tolist()
        else: rcl = ref_codes
        prompt = self._format_prompt(rcl, ref_text, chunk)
        payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048, "temperature": temperature, "top_k": top_k, "stop": ["<|SPEECH_GENERATION_END|>"], "stream": False}
        async with session.post(f"{self.api_base}/chat/completions", json=payload, timeout=60) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return self._decode(data["choices"][0]["message"]["content"])

    async def infer_batch_async(self, texts: list, ref_audio=None, ref_codes=None, ref_text=None, max_chars=256, silence_p=0.15, crossfade_p=0.0, voice=None, temperature=1.0, top_k=50, concurrency_limit=50, skip_normalize=False) -> list:
        import aiohttp
        if not skip_normalize: texts = [self.normalizer.normalize(t) for t in texts]
        if voice: ref_codes, ref_text = voice.get('codes', ref_codes), voice.get('text', ref_text)
        elif ref_audio is not None and ref_codes is None: ref_codes = self.encode_reference(ref_audio)
        elif self._default_voice and (ref_codes is None or ref_text is None):
            v = self.get_preset_voice(None); ref_codes, ref_text = v['codes'], v['text']
        if ref_codes is None or ref_text is None: raise ValueError("Must provide voice info.")
        sem = asyncio.Semaphore(concurrency_limit)
        async with aiohttp.ClientSession() as session:
            async def bounded_infer(text):
                async with sem: return await self.infer_async(text, ref_codes=ref_codes, ref_text=ref_text, max_chars=max_chars, silence_p=silence_p, crossfade_p=crossfade_p, temperature=temperature, top_k=top_k, session=session, skip_normalize=True)
            return await asyncio.gather(*(bounded_infer(t) for t in texts))
