from .standard import VieNeuTTS
from .fast import FastVieNeuTTS
from .remote import RemoteVieNeuTTS
from .xpu import XPUVieNeuTTS

def Vieneu(mode="standard", **kwargs):
    """
    Factory function for VieNeu-TTS.

    Args:
        mode: 'standard' (CPU/GPU-GGUF), 'remote' (API/Remote), 'fast' (LMDeploy), 'xpu' (Intel XPU)
        **kwargs: Arguments for chosen class

    Returns:
        VieNeuTTS | RemoteVieNeuTTS | FastVieNeuTTS | XPUVieNeuTTS instance
    """
    match mode:
        case "remote" | "api":
            return RemoteVieNeuTTS(**kwargs)
        case "fast":
            return FastVieNeuTTS(**kwargs)
        case "xpu":
            return XPUVieNeuTTS(**kwargs)
        case _:
            return VieNeuTTS(**kwargs)

__all__ = ["VieNeuTTS", "FastVieNeuTTS", "RemoteVieNeuTTS", "XPUVieNeuTTS", "Vieneu"]
