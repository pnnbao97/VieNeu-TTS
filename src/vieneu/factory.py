from .fast import FastVieNeuTTS
from .remote import RemoteVieNeuTTS
from .standard import VieNeuTTS


def Vieneu(mode="standard", **kwargs):
    """
    Factory function for VieNeu-TTS.

    Args:
        mode: 'standard' (CPU/GPU-GGUF), 'fast' (GPU-LMDeploy), 'remote' (API), 'xpu' (Intel GPU)
        **kwargs: Arguments for chosen class

    Returns:
        BaseVieneuTTS: An instance of a VieNeu-TTS implementation.
    """
    match mode:
        case "remote" | "api":
            return RemoteVieNeuTTS(**kwargs)
        case "fast" | "gpu":
            return FastVieNeuTTS(**kwargs)
        case "xpu":
            try:
                from .core_xpu import XPUVieNeuTTS
                return XPUVieNeuTTS(**kwargs)
            except Exception as e:
                raise RuntimeError(f"Failed to load XPU backend. Ensure Intel GPU drivers and torch.xpu are installed: {e}") from e
        case _:
            return VieNeuTTS(**kwargs)
