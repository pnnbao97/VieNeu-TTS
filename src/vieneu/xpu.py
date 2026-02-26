import torch
import gc
import librosa
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from neucodec import NeuCodec, DistillNeuCodec
from .standard import VieNeuTTS

class XPUVieNeuTTS(VieNeuTTS):
    """
    XPU (Intel Arc GPU) optimized implementation of VieNeu-TTS.
    """

    def __init__(
        self,
        backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
        backbone_device="xpu",
        codec_repo="neuphonic/distill-neucodec",
        codec_device="xpu",
        hf_token=None,
    ):
        if backbone_device != "xpu":
            print("Warning: XPUVieNeuTTS forced to 'xpu'.")
            backbone_device = "xpu"
        if codec_device != "xpu":
            codec_device = "xpu"

        super().__init__(
            backbone_repo=backbone_repo,
            backbone_device=backbone_device,
            codec_repo=codec_repo,
            codec_device=codec_device,
            hf_token=hf_token
        )

    def _load_backbone(self, backbone_repo, backbone_device, hf_token=None):
        print(f"Loading backbone from: {backbone_repo} on {backbone_device} (XPU) ...")
        if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
            raise RuntimeError("XPU device not available.")

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo, token=hf_token)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_repo, token=hf_token, dtype=torch.bfloat16
        ).to(device="xpu")
        print(f"   ✅ Model loaded on XPU")

    def _load_codec(self, codec_repo, codec_device):
        print(f"Loading codec from: {codec_repo} on {codec_device} (XPU) ...")
        if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
            raise RuntimeError("XPU device not available.")

        match codec_repo:
            case "neuphonic/neucodec":
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(device="xpu", dtype=torch.float32)
            case "neuphonic/distill-neucodec":
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(device="xpu", dtype=torch.float32)
            case _:
                raise ValueError(f"Unsupported codec: {codec_repo}")
        print(f"   ✅ Codec loaded on XPU")

    def _infer_torch(self, prompt_ids: list, temperature: float = 1.0, top_k: int = 50) -> str:
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to("xpu")
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

        with torch.no_grad():
            with torch.autocast(device_type="xpu", dtype=torch.bfloat16, enabled=True):
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
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        return output_str

    def encode_reference(self, ref_audio_path):
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device="xpu", dtype=torch.float32)
        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes

    def close(self):
        super().close()
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()

    def infer_batch(self, texts: list, voice=None, ref_codes=None, ref_text=None, temperature=1.0, top_k=50, skip_normalize=False) -> list:
        if voice: ref_codes, ref_text = voice.get('codes', ref_codes), voice.get('text', ref_text)
        if ref_codes is None or ref_text is None: raise ValueError("Must provide voice info.")
        if not skip_normalize: texts = [self.normalizer.normalize(t) for t in texts]

        batch_prompt_ids = [torch.tensor(self._apply_chat_template(ref_codes, ref_text, t)) for t in texts]
        inputs = self.tokenizer.pad({"input_ids": batch_prompt_ids}, padding=True, return_tensors="pt").to(device="xpu")
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

        with torch.no_grad():
            output_tokens = self.backbone.generate(**inputs, max_length=self.max_context, eos_token_id=speech_end_id, do_sample=True, temperature=temperature, top_k=top_k, use_cache=True, min_new_tokens=50)

        results = []
        input_length = inputs["input_ids"].shape[-1]
        for i in range(len(texts)):
            output_str = self.tokenizer.decode(output_tokens[i, input_length:], add_special_tokens=False)
            wav = self._decode(output_str)
            if self.watermarker: wav = self.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
            results.append(wav)

        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        return results
