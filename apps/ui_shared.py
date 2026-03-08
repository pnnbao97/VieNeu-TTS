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

# --- SHARED UI HELPERS ---

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

# --- SHARED UI LOGIC ---

class TTSAppState:
    def __init__(self):
        self.tts = None
        self.current_backbone = None
        self.current_codec = None
        self.model_loaded = False
        self.using_lmdeploy = False

def build_common_ui(mode='standard'):
    state = TTSAppState()

    def get_available_devices():
        if mode == 'xpu': return ["XPU"]
        devices = ["Auto", "CPU"]
        if sys.platform == "darwin":
            if torch.backends.mps.is_available(): devices.append("MPS")
        elif torch.cuda.is_available(): devices.append("CUDA")
        return devices

    def get_status_msg():
        if not state.model_loaded or state.tts is None: return "⏳ Chưa tải model."
        backend = "🚀 LMDeploy" if state.using_lmdeploy else ("🚀 Intel XPU" if mode == 'xpu' else "📦 Standard")
        return f"✅ Model đã tải thành công!\n\n🔧 Backend: {backend}\n🦜 Backbone: {state.current_backbone}\n🎵 Codec: {state.current_codec}"

    def load_model(backbone_choice, codec_choice, device_choice, force_lmdeploy, custom_id="", custom_base="", custom_token=""):
        state.model_loaded = False
        yield "⏳ Đang tải model...", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        try:
            if state.tts is not None: state.tts = None; cleanup_gpu_memory()
            config, custom_load, is_lora = get_backbone_config(backbone_choice, custom_id, custom_base)
            codec_cfg = CODEC_CONFIGS[codec_choice]

            if mode == 'xpu':
                from vieneu.core_xpu import XPUVieNeuTTS
                state.tts = XPUVieNeuTTS(backbone_repo=config["repo"], codec_repo=codec_cfg["repo"], hf_token=custom_token)
                state.using_lmdeploy = False
            else:
                use_lmdeploy = force_lmdeploy and (torch.cuda.is_available() if device_choice in ["Auto", "CUDA"] else False) and "gguf" not in config["repo"].lower()
                if use_lmdeploy:
                    target = config["repo"]
                    if custom_load and is_lora:
                        safe = custom_id.strip().replace("/", "_").replace("\\", "_")
                        cache = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "merged_models_cache", safe))
                        target = cache
                        if not os.path.exists(os.path.join(cache, "vocab.json")):
                            yield "⏳ Merging LoRA...", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                            temp = VieNeuTTS(BACKBONE_CONFIGS[custom_base]["repo"], hf_token=custom_token)
                            temp.load_lora_adapter(custom_id.strip(), hf_token=custom_token)
                            if hasattr(temp.backbone, "merge_and_unload"): temp.backbone = temp.backbone.merge_and_unload()
                            temp.backbone.save_pretrained(cache); temp.tokenizer.save_pretrained(cache)
                            import json
                            with open(os.path.join(cache, "voices.json"), "w", encoding="utf-8") as f:
                                json.dump({"default_voice": temp._default_voice, "presets": temp._preset_voices}, f)
                            del temp; cleanup_gpu_memory()
                    state.tts = FastVieNeuTTS(backbone_repo=target, codec_repo=codec_cfg["repo"], hf_token=custom_token)
                    state.using_lmdeploy = True
                else:
                    dev = "mps" if sys.platform == "darwin" and device_choice == "Auto" else device_choice.lower()
                    if "gguf" in config["repo"].lower(): dev = "gpu"
                    state.tts = VieNeuTTS(backbone_repo=config["repo"], backbone_device=dev, codec_repo=codec_cfg["repo"], hf_token=custom_token)
                    state.using_lmdeploy = False

            if is_lora and not state.using_lmdeploy:
                state.tts.load_lora_adapter(custom_id.strip(), hf_token=custom_token)
                if hasattr(state.tts.backbone, "merge_and_unload"): state.tts.backbone = state.tts.backbone.merge_and_unload()

            state.current_backbone, state.current_codec, state.model_loaded = backbone_choice, codec_choice, True
            v = state.tts.list_preset_voices() or []
            yield get_status_msg(), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(choices=v, value=state.tts._default_voice if v else None, interactive=True), gr.update(visible=True), gr.update(visible=True), gr.update(selected="preset_mode"), "preset_mode"
        except Exception as e:
            yield f"❌ Lỗi: {e}", gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    def synthesize(text, voice, custom_audio, custom_text, mode_tab, gen_mode, use_batch, batch_size, temp, max_chars):
        if not state.model_loaded: yield None, "⚠️ Tải model trước!"; return
        try:
            ref_codes, ref_text = resolve_reference_voice(state.tts, mode_tab, voice, custom_audio, custom_text)
            if isinstance(ref_codes, torch.Tensor): ref_codes = ref_codes.cpu().numpy()
        except Exception as e: yield None, f"❌ Lỗi: {e}"; return

        if gen_mode == "Standard (Một lần)":
            chunks = split_text_into_chunks(_text_normalizer.normalize(text.strip()), max_chars=max_chars)
            all_wavs = []
            try:
                if use_batch and hasattr(state.tts, "infer_batch") and len(chunks) > 1:
                    all_wavs = [w for w in state.tts.infer_batch(chunks, ref_codes=ref_codes, ref_text=ref_text, max_batch_size=batch_size, temperature=temp, skip_normalize=True) if w is not None]
                else:
                    for c in chunks:
                        w = state.tts.infer(c, ref_codes=ref_codes, ref_text=ref_text, temperature=temp, skip_normalize=True)
                        if w is not None: all_wavs.append(w)
                if not all_wavs: yield None, "❌ Lỗi sinh audio"; return
                yield wrap_tempfile_sf_write(join_audio_chunks(all_wavs, sr=24000), 24000), "✅ Hoàn tất!"
            except Exception as e: yield None, f"❌ Lỗi: {e}"
            finally: cleanup_gpu_memory()
        else:
            sr, q, end_ev, err_ev = 24000, queue.Queue(), threading.Event(), threading.Event()
            def producer():
                try:
                    for c in split_text_into_chunks(_text_normalizer.normalize(text.strip()), max_chars=max_chars):
                        for p in state.tts.infer_stream(c, ref_codes=ref_codes, ref_text=ref_text, temperature=temp, skip_normalize=True):
                            if p is not None: q.put((sr, p))
                except Exception as ex: err_ev.set(); print(ex)
                finally: end_ev.set(); q.put(None)
            threading.Thread(target=producer, daemon=True).start()
            full = []
            while not (end_ev.is_set() and q.empty()):
                try:
                    item = q.get(timeout=0.1)
                    if item: full.append(item[1]); yield item, "🔊 Đang phát..."
                except:
                    if err_ev.is_set(): yield None, "❌ Lỗi"; return
            if full: yield wrap_tempfile_sf_write(np.concatenate(full), sr), "✅ Hoàn tất!"

    title = "VieNeu-TTS" + (" (XPU)" if mode == 'xpu' else "")
    with gr.Blocks(theme=UI_THEME, css=COMMON_CSS, title=title, head=FAVICON_HEAD) as demo:
        with gr.Column(elem_classes="container"):
            gr.HTML(HEADER_HTML.replace("Studio", "Studio (XPU Edition)") if mode == 'xpu' else HEADER_HTML)
            with gr.Group():
                with gr.Row():
                    backbone_select = gr.Dropdown(list(BACKBONE_CONFIGS.keys()) + ["Custom Model"], value="VieNeu-TTS (GPU)", label="🦜 Backbone")
                    codec_select = gr.Dropdown(list(CODEC_CONFIGS.keys()), value="NeuCodec (Distill)", label="🎵 Codec")
                    device_choice = gr.Radio(get_available_devices(), value=get_available_devices()[0], label="🖥️ Device")
                with gr.Row(visible=False) as custom_group:
                    custom_id = gr.Textbox(label="📦 Custom Model ID", scale=2)
                    custom_token = gr.Textbox(label="🔑 HF Token", type="password", scale=1)
                    custom_base = gr.Dropdown([k for k in BACKBONE_CONFIGS.keys() if "gguf" not in k.lower()], label="🔗 Base Model", value="VieNeu-TTS-0.3B (GPU)", visible=False, scale=1)
                force_lmdeploy = gr.Checkbox(value=True, label="🚀 Optimize with LMDeploy", visible=(mode != 'xpu'))
                btn_load = gr.Button("🔄 Tải Model", variant="primary"); model_status = gr.Markdown("⏳ Chưa tải model.")
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(label="Văn bản", lines=4, value="Hà Nội, trái tim của Việt Nam...")
                    with gr.Tabs() as tabs:
                        with gr.TabItem("👤 Preset", id="preset_mode") as tab_preset: voice_select = gr.Dropdown(choices=[], label="Giọng mẫu")
                        with gr.TabItem("🦜 Voice Cloning", id="custom_mode") as tab_custom:
                            custom_audio, custom_text = gr.Audio(label="Audio mẫu", type="filepath"), gr.Textbox(label="Nội dung audio mẫu")
                            cloning_warning = gr.Markdown(visible=False)
                    gen_radio = gr.Radio(["Standard (Một lần)", "Streaming (Phát trực tiếp)"], value="Standard (Một lần)", label="Chế độ sinh")
                    use_batch, batch_size_slider = gr.Checkbox(value=True, label="⚡ Batch Processing"), gr.Slider(1, 256 if mode == 'xpu' else 16, 128 if mode == 'xpu' else 4, step=1, label="📊 Batch Size")
                    with gr.Accordion("⚙️ Nâng cao", open=False):
                        temp_slider, max_chars_slider = gr.Slider(0.1, 1.5, 1.0, step=0.1, label="🌡️ Temp"), gr.Slider(64 if mode == 'xpu' else 128, 512, 128 if mode == 'xpu' else 256, step=16 if mode == 'xpu' else 32, label="📝 Max Chars")
                    current_tab_state = gr.State("preset_mode")
                    with gr.Row(): btn_generate, btn_stop = gr.Button("🎵 Bắt đầu", variant="primary", scale=2, interactive=False), gr.Button("⏹️ Dừng", variant="stop", scale=1, interactive=False)
                with gr.Column(scale=2): audio_output, status_output = gr.Audio(label="Kết quả", type="filepath", autoplay=True), gr.Textbox(label="Trạng thái", elem_classes="status-box")

            codec_select.change(on_codec_change, [codec_select, current_tab_state], [tab_custom, tabs, current_tab_state])
            tab_preset.select(lambda: "preset_mode", None, current_tab_state); tab_custom.select(lambda: "custom_mode", None, current_tab_state)
            custom_audio.change(validate_audio_duration, [custom_audio], [cloning_warning])
            backbone_select.change(on_backbone_change, [backbone_select], [custom_group])
            custom_id.change(on_custom_id_change, [custom_id], [custom_base])
            btn_load.click(load_model, [backbone_select, codec_select, device_choice, force_lmdeploy, custom_id, custom_base, custom_token], [model_status, btn_generate, btn_load, btn_stop, voice_select, tab_preset, tab_custom, tabs, current_tab_state])
            gen_ev = btn_generate.click(synthesize, [text_input, voice_select, custom_audio, custom_text, current_tab_state, gen_radio, use_batch, batch_size_slider, temp_slider, max_chars_slider], [audio_output, status_output])
            btn_generate.click(lambda: gr.update(interactive=True), None, btn_stop)
            gen_ev.then(lambda: gr.update(interactive=False), None, btn_stop); btn_stop.click(None, None, None, cancels=[gen_ev])
            demo.load(lambda: (get_status_msg(), gr.update(interactive=state.model_loaded), gr.update(interactive=False)), None, [model_status, btn_generate, btn_stop])
    return demo
