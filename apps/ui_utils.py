import time
import gc
import sys
import gradio as gr
import soundfile as sf
from functools import lru_cache

def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

def _split_estimate_status(status: str) -> tuple[str, str]:
    if not isinstance(status, str):
        return status, ""

    estimate_marker = " | Ước tính còn lại: "
    if estimate_marker in status:
        status_text, estimate_text = status.split(" | ", 1)
        if status.endswith("...") and not status_text.endswith("..."):
            status_text += "..."
        return status_text, estimate_text.rstrip(". ")

    if ("batch mẫu:" in status or "trung bình batch:" in status) and "ước tính còn lại:" in status:
        start = status.find("(")
        end = status.rfind(")")
        if start != -1 and end != -1 and end > start:
            status_text = status[:start].strip()
            estimate_text = status[start + 1:end].replace(", ", "\n")
            return status_text, estimate_text

    return status, ""

def _extract_progress(status: str) -> tuple[str, int, int] | None:
    if not isinstance(status, str):
        return None

    for marker, label in (("Đang xử lý batch ", "batch"), ("Đang xử lý đoạn ", "đoạn")):
        if marker not in status:
            continue

        progress_text = status.split(marker, 1)[1].split(" ", 1)[0].strip(".")
        if "/" not in progress_text:
            return None

        current_text, total_text = progress_text.split("/", 1)
        try:
            current = int(current_text)
            total = int(total_text)
        except ValueError:
            return None

        if current > 0 and total > 0:
            return label, current, total

    return None

def wrap_with_estimate(synthesize_fn):
    def wrapper(*args):
        previous_progress_time = None
        total_unit_duration = 0.0
        completed_units = 0

        for audio_path, status in synthesize_fn(*args):
            status_text, estimate_text = _split_estimate_status(status)

            if not estimate_text:
                progress = _extract_progress(status_text)
                if progress:
                    unit_label, current, total = progress
                    now = time.time()
                    if previous_progress_time is not None:
                        total_unit_duration += now - previous_progress_time
                        completed_units += 1
                    previous_progress_time = now

                    if completed_units == 0:
                        estimate_text = f"Đang đo thời gian {unit_label} đầu tiên..."
                    else:
                        average_unit_duration = total_unit_duration / completed_units
                        estimated_total = average_unit_duration * total
                        estimated_remaining = average_unit_duration * max(0, total - current + 1)
                        estimate_text = (
                            f"Ước tính còn lại: {_format_duration(estimated_remaining)}\n"
                            f"Tổng: {_format_duration(estimated_total)}"
                        )

            yield audio_path, status_text, estimate_text
    return wrapper

def cleanup_gpu_memory():
    """Aggressively cleanup GPU memory (CUDA, MPS, XPU)"""
    if 'torch' in sys.modules:
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            # Another tab may be capturing a v3 Turbo CUDA graph: a sync or
            # cache release in the middle would break it (see capture_lock.py).
            from vieneu._v3_turbo_engine.capture_lock import CAPTURE_LOCK
            with CAPTURE_LOCK.shared():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
    gc.collect()

@lru_cache(maxsize=32)
def get_ref_text_cached(text_path: str) -> str:
    """Cache reference text loading"""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()

def on_codec_change(codec: str, current_mode: str):
    is_onnx = "onnx" in codec.lower()
    if is_onnx and current_mode == "custom_mode":
        return gr.update(visible=False), gr.update(selected="preset_mode"), "preset_mode"
    return gr.update(visible=not is_onnx), gr.update(), current_mode

def validate_audio_duration(audio_path):
    if not audio_path:
        return gr.update(visible=False)
    try:
        info = sf.info(audio_path)
        if info.duration > 5.1:
            return gr.update(
                value=f"⚠️ **Cảnh báo:** Audio mẫu hiện tại dài {info.duration:.1f} giây. Để có kết quả clone giọng tối ưu, bạn nên sử dụng đoạn audio có độ dài lý tưởng từ **3 đến 5 giây**.",
                visible=True
            )
    except Exception:
        pass
    return gr.update(visible=False)

def on_custom_id_change(model_id):
    # Auto detect LoRA and base model
    if model_id and "lora" in model_id.lower():
        # Detect base model
        if "0.3" in model_id:
            base_model = "VieNeu-TTS-0.3B (GPU)"
        else:
            base_model = "VieNeu-TTS (GPU)"

        return (
            gr.update(visible=True, value=base_model),
            gr.update(), gr.update()
        )

    return (
        gr.update(visible=False),
        gr.update(),
        gr.update()
    )
