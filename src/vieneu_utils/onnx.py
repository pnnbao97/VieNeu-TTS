import logging
from typing import List, Optional, Sequence

logger = logging.getLogger("Vieneu.ONNX")


def select_onnx_providers(
    available_providers: Sequence[str],
    device: str = "cpu",
    requested_providers: Optional[Sequence[str]] = None,
) -> List[str]:
    """Select installed ONNX Runtime providers in deterministic priority order.

    ``requested_providers`` gives callers full control while still failing early
    when the installed ONNX Runtime package cannot satisfy their request. With
    no override, a provider is selected from the requested device and the CPU
    provider is retained as a fallback when available.
    """
    available = list(available_providers)
    available_set = set(available)

    if requested_providers is not None:
        requested = list(requested_providers)
        if not requested:
            raise ValueError("onnx_providers must contain at least one provider")
        missing = [provider for provider in requested if provider not in available_set]
        if missing:
            raise ValueError(
                "Requested ONNX Runtime provider(s) are unavailable: "
                f"{', '.join(missing)}. Available providers: {', '.join(available)}"
            )
        return requested

    device_name = str(device or "cpu").lower().split(":", 1)[0]
    provider_preferences = {
        "cuda": ["CUDAExecutionProvider"],
        "gpu": ["CUDAExecutionProvider"],
        "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        "trt": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        "rocm": ["ROCMExecutionProvider", "MIGraphXExecutionProvider"],
        "amd": ["MIGraphXExecutionProvider", "ROCMExecutionProvider"],
        "xpu": ["OpenVINOExecutionProvider"],
        "openvino": ["OpenVINOExecutionProvider"],
        "mps": ["CoreMLExecutionProvider"],
        "coreml": ["CoreMLExecutionProvider"],
        "dml": ["DmlExecutionProvider"],
        "directml": ["DmlExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }
    selected = [
        provider
        for provider in provider_preferences.get(device_name, [])
        if provider in available_set
    ]
    if "CPUExecutionProvider" in available_set and "CPUExecutionProvider" not in selected:
        selected.append("CPUExecutionProvider")
    if selected:
        return selected
    if available:
        logger.warning(
            "No preferred ONNX Runtime provider is available for device '%s'; "
            "falling back to %s",
            device,
            available[0],
        )
        return [available[0]]
    raise RuntimeError("ONNX Runtime reported no available execution providers")
