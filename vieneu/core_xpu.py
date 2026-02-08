import torch
import gc
import librosa
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from neucodec import NeuCodec, DistillNeuCodec
from .core import VieNeuTTS


class XPUVieNeuTTS(VieNeuTTS):
    """
    XPU (Intel Arc GPU) optimized implementation of VieNeu-TTS.
    Uses native PyTorch XPU backend with bfloat16 and specialized compilation/warmup.
    """

    def __init__(
        self,
        backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
        backbone_device="xpu", # Forced default
        codec_repo="neuphonic/distill-neucodec",
        codec_device="xpu",    # Forced default
        hf_token=None,
    ):
        # Ensure we are strictly on XPU
        if backbone_device != "xpu":
            print("Warning: XPUVieNeuTTS initialized with non-xpu device. Forcing 'xpu'.")
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
        """XPU (Intel Arc GPU) loading implementation using native PyTorch XPU."""
        print(f"Loading backbone from: {backbone_repo} on {backbone_device} (XPU) ...")
        
        # Verify XPU is available
        if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
            raise RuntimeError("XPU device requested but torch.xpu.is_available() returned False")
        
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo, token=hf_token)
        self.tokenizer.padding_side = "left"
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load in bfloat16 for XPU optimization
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_repo, 
            token=hf_token, 
            dtype=torch.bfloat16
        ).to(device="xpu")
        
        print(f"   ✅ Model loaded on XPU device")

    def _load_codec(self, codec_repo, codec_device):
        """XPU (Intel Arc GPU) codec loading implementation."""
        print(f"Loading codec from: {codec_repo} on {codec_device} (XPU) ...")
        
        if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
            raise RuntimeError("XPU device requested but torch.xpu.is_available() returned False")
        
        match codec_repo:
            case "neuphonic/neucodec":
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(device="xpu", dtype=torch.float32)
            case "neuphonic/distill-neucodec":
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(device="xpu", dtype=torch.float32)
            case "neuphonic/neucodec-onnx-decoder-int8":
                raise ValueError("ONNX decoder does not support XPU device. Use CPU codec.")
            case _:
                raise ValueError(f"Unsupported codec repository: {codec_repo}")
        
        print(f"   ✅ Codec loaded on XPU device")



    def _infer_torch(self, prompt_ids: list[int], temperature: float = 1.0, top_k: int = 50) -> str:
        """XPU-specific inference using native PyTorch XPU with autocast."""
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to("xpu")
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        
        with torch.no_grad():
            # Use XPU autocast for performance
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
        output_str = self.tokenizer.decode(
            output_tokens[0, input_length:].cpu().numpy().tolist(), add_special_tokens=False
        )
        
        # Cleanup XPU memory after generation
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        
        return output_str

    def encode_reference(self, ref_audio_path):
        """Override to ensure input tensor is on XPU."""
        
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
        
        # Move to XPU explicitly
        wav_tensor = wav_tensor.to(device="xpu", dtype=torch.float32)
        
        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes
        
    def close(self):
        """Extended close to handle XPU cache clearing."""
        super().close()
        try:
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                torch.xpu.empty_cache()
        except:
            pass
    
    def infer_batch(
            self, 
            texts: list[str], 
            voice: dict = None, 
            ref_codes: torch.Tensor = None, 
            ref_text: str = None,
            temperature: float = 1.0, 
            top_k: int = 50
            ) -> list[np.ndarray]:
        """
        Thực hiện inference theo batch trên XPU sử dụng thuần PyTorch.
        """
        if voice is not None:
            ref_codes = voice.get('codes', ref_codes)
            ref_text = voice.get('text', ref_text)
        
        if ref_codes is None or ref_text is None:
            raise ValueError("Phải cung cấp voice hoặc ref_codes và ref_text.")

        # Prepare prompt for each chunk in batch
        batch_prompt_ids = []
        for text in texts:
            prompt_ids = self._apply_chat_template(ref_codes, ref_text, text)
            batch_prompt_ids.append(torch.tensor(prompt_ids))
            
        inputs = self.tokenizer.pad(
            {"input_ids": batch_prompt_ids}, 
            padding=True, 
            return_tensors="pt"
        ).to(device="xpu")

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

        # Batch Decoding
        results = []
        input_length = inputs["input_ids"].shape[-1]
        
        for i in range(len(texts)):
            generated_ids = output_tokens[i, input_length:]
            output_str = self.tokenizer.decode(generated_ids, add_special_tokens=False)
            wav = self._decode(output_str)
            
            if self.watermarker:
                wav = self.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
                
            results.append(wav)

        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        
        return results