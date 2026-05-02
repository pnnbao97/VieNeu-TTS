

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
            from .remote import RemoteVieNeuTTS
            return RemoteVieNeuTTS(**kwargs)
        case "fast" | "gpu":
            from .fast import FastVieNeuTTS
            return FastVieNeuTTS(**kwargs)
        case "turbo":
            from .turbo import TurboVieNeuTTS
            return TurboVieNeuTTS(**kwargs)
        case "turbo_gpu":
            from .turbo import TurboGPUVieNeuTTS
            return TurboGPUVieNeuTTS(**kwargs)
        case "xpu":
            try:
                from .core_xpu import XPUVieNeuTTS
                return XPUVieNeuTTS(**kwargs)
            except Exception as e:
                raise RuntimeError(f"Failed to load XPU backend. Ensure Intel GPU drivers and torch.xpu are installed: {e}") from e
        case "standard":
            from .standard import VieNeuTTS
            return VieNeuTTS(**kwargs)
