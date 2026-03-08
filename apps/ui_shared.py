import os
import yaml
import torch
import gc
import soundfile as sf
import gradio as gr
import tempfile
import time
import numpy as np
import queue
import threading
from vieneu_utils.core_utils import env_bool, split_text_into_chunks, join_audio_chunks
from vieneu_utils.normalize_text import VietnameseTTSNormalizer
from vieneu import VieNeuTTS, FastVieNeuTTS
from functools import lru_cache

# --- CONSTANTS & CONFIG ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

def load_config():
    try:
        if not os.path.exists(CONFIG_PATH):
            return {}
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️ Warning: Could not read config.yaml: {e}")
        return {}

_config = load_config()
BACKBONE_CONFIGS = _config.get("backbone_configs", {})
CODEC_CONFIGS = _config.get("codec_configs", {})
_text_settings = _config.get("text_settings", {})
MAX_CHARS_PER_CHUNK = _text_settings.get("max_chars_per_chunk", 256)

_text_normalizer = VietnameseTTSNormalizer()

@lru_cache(maxsize=32)
def get_ref_text_cached(text_path: str) -> str:
    """Cache reference text loading"""
    if not os.path.exists(text_path):
        return ""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()

def cleanup_gpu_memory():
    """Aggressively cleanup GPU/XPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
    gc.collect()

def wrap_tempfile_sf_write(wav, sr):
    """Securely write audio to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=tempfile.gettempdir()) as tmp:
        sf.write(tmp.name, wav, sr)
        return tmp.name

# --- SHARED UI LOGIC ---

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
                value=f"⚠️ **Cảnh báo:** Audio mẫu dài {info.duration:.1f}s. Lý tưởng nhất là 3-5s.",
                visible=True,
            )
    except Exception:
        pass
    return gr.update(visible=False)

def on_backbone_change(choice):
    return gr.update(visible=choice == "Custom Model")

def on_custom_id_change(model_id):
    if model_id and "lora" in model_id.lower():
        base = "VieNeu-TTS-0.3B (GPU)" if "0.3" in model_id else "VieNeu-TTS (GPU)"
        return gr.update(visible=True, value=base)
    return gr.update(visible=False)

def get_backbone_config(backbone_choice, custom_model_id, custom_base_model):
    if backbone_choice == "Custom Model":
        if not custom_model_id or not custom_model_id.strip():
            raise ValueError("Vui lòng nhập Model ID cho Custom Model.")

        # Basic validation
        if ".." in custom_model_id or (custom_model_id.startswith("/") and not os.path.exists(custom_model_id)):
             raise ValueError("Model ID không hợp lệ.")

        if "lora" in custom_model_id.lower():
            if custom_base_model not in BACKBONE_CONFIGS:
                raise ValueError(f"Base Model '{custom_base_model}' không hợp lệ.")
            base_config = BACKBONE_CONFIGS[custom_base_model]
            return {
                "repo": base_config["repo"],
                "supports_streaming": base_config["supports_streaming"],
                "description": f"Custom Merged: {custom_model_id} + {custom_base_model}",
            }, True, True
        return {
            "repo": custom_model_id.strip(),
            "supports_streaming": False,
            "description": f"Custom Model: {custom_model_id}",
        }, True, False
    return BACKBONE_CONFIGS[backbone_choice], False, False

def resolve_reference_voice(tts, mode_tab, voice_choice, custom_audio, custom_text):
    if mode_tab == "preset_mode":
        if not voice_choice or "⚠️" in voice_choice:
            raise ValueError("Vui lòng chọn giọng mẫu.")
        voice_data = tts.get_preset_voice(voice_choice)
        return voice_data["codes"], voice_data["text"]
    if mode_tab == "custom_mode":
        if custom_audio is None: raise ValueError("Vui lòng upload Audio mẫu!")
        if not custom_text or not custom_text.strip(): raise ValueError("Vui lòng nhập Reference Text!")
        return tts.encode_reference(custom_audio), custom_text.strip()
    raise ValueError(f"Unknown mode: {mode_tab}")

# --- CSS & HTML ---

UI_THEME = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="cyan", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
).set(
    button_primary_background_fill="linear-gradient(90deg, #6366f1 0%, #0ea5e9 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4f46e5 0%, #0284c7 100%)",
)

COMMON_CSS = """
.container { max-width: 1400px; margin: auto; }
.header-box {
    text-align: center; margin-bottom: 25px; padding: 25px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px; color: white !important;
}
.header-title { font-size: 2.5rem; font-weight: 800; color: white !important; }
.gradient-text { background: -webkit-linear-gradient(45deg, #60A5FA, #22D3EE); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.header-icon { color: white; }
.status-box { font-weight: 500; border: 1px solid rgba(99, 102, 241, 0.1); background: rgba(99, 102, 241, 0.03); border-radius: 8px; }
.model-card-content { display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; font-size: 0.9rem; color: white !important; }
.model-card-link { color: #60A5FA; text-decoration: none; font-weight: 500; }
.warning-banner { background-color: #fffbeb; border: 1px solid #fef3c7; border-radius: 12px; padding: 16px; margin-bottom: 20px; }
"""

HEADER_HTML = """
<div class="header-box">
    <h1 class="header-title"><span class="header-icon">🦜</span><span class="gradient-text">VieNeu-TTS Studio</span></h1>
    <div class="model-card-content">
        <strong>Models:</strong>
        <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS" target="_blank" class="model-card-link">VieNeu-TTS</a> |
        <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B" target="_blank" class="model-card-link">VieNeu-TTS-0.3B</a> |
        <a href="https://github.com/pnnbao97/VieNeu-TTS" target="_blank" class="model-card-link">GitHub</a>
    </div>
</div>
"""

FAVICON_HEAD = """<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🦜</text></svg>">"""
