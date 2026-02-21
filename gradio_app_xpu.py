import os
import sys
import subprocess

# --- Add XPU dll path ---
intel_dll_path = os.path.join(sys.prefix, 'Library', 'bin')

if os.path.exists(intel_dll_path):
    os.environ['PATH'] = intel_dll_path + os.pathsep + os.environ['PATH']
else:
    print(f"⚠️ Không tìm thấy thư mục DLL Intel XPU: {intel_dll_path}")
    
import torch
import gradio as gr
import soundfile as sf
import tempfile
import time
import numpy as np
import queue
import threading
import yaml
import gc

from vieneu.core_xpu import XPUVieNeuTTS
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks, env_bool
from vieneu_utils.normalize_text import VietnameseTTSNormalizer
from functools import lru_cache

try:
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        print("⚠️ Không tìm thấy thiết bị Intel XPU (Intel Arc GPU).")
        print("🔄 Đang tự động chuyển hướng sang phiên bản CPU/CUDA (gradio_app.py)...")
        # Chạy file gradio_app.py và truyền tiếp các arguments (nếu có)
        subprocess.run([sys.executable, "gradio_app.py"] + sys.argv[1:])
        sys.exit(0)
except ImportError:
    pass

print("⏳ Đang khởi động VieNeu-TTS (Phiên bản tối ưu cho Intel XPU)...")


# --- CONSTANTS & CONFIG ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}
except Exception as e:
    raise RuntimeError(f"Không thể đọc config.yaml: {e}")

BACKBONE_CONFIGS = _config.get("backbone_configs", {})
CODEC_CONFIGS = _config.get("codec_configs", {})

_text_settings = _config.get("text_settings", {})
MAX_CHARS_PER_CHUNK = _text_settings.get("max_chars_per_chunk", 256)

if not BACKBONE_CONFIGS or not CODEC_CONFIGS:
    raise ValueError("config.yaml thiếu backbone_configs hoặc codec_configs")

# --- 1. MODEL CONFIGURATION ---
# Global model instance
tts = None
current_backbone = None
current_codec = None
model_loaded = False

# Normalizer (module-level singleton)
_text_normalizer = VietnameseTTSNormalizer()

def get_available_devices() -> list[str]:
    """Chỉ trả về XPU cho phiên bản này."""
    return ["XPU"]

def get_model_status_message() -> str:
    """Reconstruct status message from global state"""
    global model_loaded, tts, current_backbone, current_codec
    if not model_loaded or tts is None:
        return "⏳ Chưa tải model."
    
    backbone_config = BACKBONE_CONFIGS.get(current_backbone, {})
    codec_config = CODEC_CONFIGS.get(current_codec, {})
    
    backend_name = "🚀 Intel XPU (BFloat16 native)"
    codec_device = "CPU" if "ONNX" in (current_codec or "") else "XPU"
    
    preencoded_note = "\n⚠️ Codec ONNX không hỗ trợ chức năng clone giọng nói." if codec_config.get('use_preencoded') else ""

    return (
        f"✅ Model đã tải thành công!\n\n"
        f"🔧 Backend: {backend_name}\n"
        f"🦜 Backbone: {current_backbone}\n"
        f"🎵 Codec: {current_codec}\n"
        f"🖥️ Thiết bị chạy Codec: {codec_device}{preencoded_note}"
    )

def restore_ui_state():
    """Update UI components based on persistence"""
    global model_loaded
    msg = get_model_status_message()
    return (
        msg, 
        gr.update(interactive=model_loaded), # btn_generate
        gr.update(interactive=False)         # btn_stop
    )

@lru_cache(maxsize=32)
def get_ref_text_cached(text_path: str) -> str:
    """Cache reference text loading"""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()

def cleanup_gpu_memory():
    """Aggressively cleanup XPU memory"""
    if torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
    gc.collect()

def load_model(backbone_choice: str, codec_choice: str, device_choice: str, 
               custom_model_id: str = "", custom_base_model: str = "", custom_hf_token: str = ""):
    """Load model with XPU optimizations"""
    global tts, current_backbone, current_codec, model_loaded
    model_loaded = False 
    
    yield (
        "⏳ Đang tải model lên Intel XPU... Vui lòng chờ trong giây lát...",
        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    )
    
    try:
        # Cleanup before loading new model
        if tts is not None:
            tts = None 
            cleanup_gpu_memory()
        
        custom_loading = False
        is_merged_lora = False

        if backbone_choice == "Custom Model":
            custom_loading = True
            if not custom_model_id or not custom_model_id.strip():
                yield (
                    "❌ Lỗi: Vui lòng nhập Model ID cho Custom Model.",
                    gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                )
                return

            if "lora" in custom_model_id.lower():
                print(f"🔄 Detected LoRA in name. preparing merge with base: {custom_base_model}")
                if custom_base_model not in BACKBONE_CONFIGS:
                    yield (
                        f"❌ Lỗi: Base Model '{custom_base_model}' không hợp lệ.",
                        gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    )
                    return
                
                base_config = BACKBONE_CONFIGS[custom_base_model]
                backbone_config = {
                    "repo": base_config["repo"],
                    "supports_streaming": base_config["supports_streaming"],
                    "description": f"Custom Merged: {custom_model_id} + {custom_base_model}"
                }
                is_merged_lora = True
            else:
                backbone_config = {
                    "repo": custom_model_id.strip(),
                    "supports_streaming": False,
                    "description": f"Custom Model: {custom_model_id}"
                }
        else:
            backbone_config = BACKBONE_CONFIGS[backbone_choice]
            
        codec_config = CODEC_CONFIGS[codec_choice]
        
        # Bắt buộc thiết lập device là XPU
        backbone_device = "xpu"
        codec_device = "cpu" if "ONNX" in codec_choice else "xpu"
        
        print(f"📦 Loading model on XPU...")
        print(f"   Backbone: {backbone_config['repo']} on {backbone_device}")
        print(f"   Codec: {codec_config['repo']} on {codec_device}")
        
        tts = XPUVieNeuTTS(
            backbone_repo=backbone_config["repo"],
            backbone_device=backbone_device,
            codec_repo=codec_config["repo"],
            codec_device=codec_device,
            hf_token=custom_hf_token
        )

        # Xử lý LoRA Merge trực tiếp trên XPU
        if is_merged_lora and custom_loading:
            yield (
                f"🔄 Đang tải và merge LoRA adapter: {custom_model_id} trên XPU...",
                gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            )
            try:
                tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                if hasattr(tts, 'backbone') and hasattr(tts.backbone, 'merge_and_unload'):
                    print("   🔄 Merging LoRA into backbone...")
                    tts.backbone = tts.backbone.merge_and_unload()
                    tts._lora_loaded = False 
                    tts._current_lora_repo = None
                    print("   ✅ Merged successfully!")
                else:
                    print("   ⚠️ Warning: Model does not support merge_and_unload, keeping adapter active.")
            except Exception as e:
                 raise RuntimeError(f"Failed to merge LoRA: {e}")
        
        current_backbone = backbone_choice
        current_codec = codec_choice
        model_loaded = True
        
        success_msg = get_model_status_message()
            
        # Prepare voice update
        try:
            voices = tts.list_preset_voices()
        except Exception:
            voices = []

        has_voices = len(voices) > 0
        
        if has_voices:
            default_v = tts._default_voice
            is_tuple = (len(voices) > 0 and isinstance(voices[0], tuple))
            voice_values = [v[1] for v in voices] if is_tuple else voices
            
            if not default_v and voice_values:
                 default_v = voice_values[0]

            if default_v and default_v not in voice_values:
                if is_tuple:
                    voices.append((default_v, default_v))
                else:
                    voices.append(default_v)
            
            if is_tuple:
                voices.sort(key=lambda x: str(x[0]))
            else:
                voices.sort()

            voice_update = gr.update(choices=voices, value=default_v, interactive=True)
            
            tab_p = gr.update(visible=True)
            tab_c = gr.update(visible=True)
            tab_sel = gr.update(selected="preset_mode")
            mode_state = "preset_mode"
        else:
            msg = "⚠️ Không tìm thấy file voices.json. Vui lòng dùng Tab Voice Cloning."
            voice_update = gr.update(choices=[msg], value=msg, interactive=False)
            
            tab_p = gr.update(visible=True)
            tab_c = gr.update(visible=True)
            tab_sel = gr.update(selected="preset_mode")
            mode_state = "preset_mode"

        yield (
            success_msg,
            gr.update(interactive=True), # btn_generate
            gr.update(interactive=True), # btn_load
            gr.update(interactive=False), # btn_stop
            voice_update,
            tab_p, tab_c, tab_sel, mode_state
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        model_loaded = False

        yield (
            f"❌ Lỗi khi tải model: {str(e)}",
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update()
        )


# --- 2. DATA & HELPERS ---

def synthesize_speech(text: str, voice_choice: str, custom_audio, custom_text: str, 
                      mode_tab: str, generation_mode: str,
                      use_batch: bool, max_batch_size_run: int, # Added as decoys
                      temperature: float, max_chars_chunk: int):
    """Synthesis using XPU logic (Sequential generation with autocast)"""
    global tts, current_backbone, current_codec, model_loaded
    
    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return
    
    if not text or text.strip() == "":
        yield None, "⚠️ Vui lòng nhập văn bản!"
        return
    
    raw_text = text.strip()
    
    yield None, "📄 Đang xử lý Reference..."
    
    try:
        ref_codes = None
        ref_text_raw = ""
        
        if mode_tab == "preset_mode":
            if not voice_choice or "⚠️" in voice_choice:
                raise ValueError("Vui lòng chọn giọng mẫu hoặc chuyển sang Tab Voice Cloning.")
            
            voice_data = tts.get_preset_voice(voice_choice)
            ref_codes = voice_data['codes']
            ref_text_raw = voice_data['text']
            
        elif mode_tab == "custom_mode":
            if custom_audio is None:
                 raise ValueError("Vui lòng upload file Audio mẫu (Reference Audio)!")
            if not custom_text or not custom_text.strip():
                 raise ValueError("Vui lòng nhập nội dung văn bản của Audio mẫu (Reference Text)!")
            
            ref_text_raw = custom_text.strip()
            ref_codes = tts.encode_reference(custom_audio)
            
        else:
            raise ValueError(f"Unknown mode: {mode_tab}")

        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()

    except Exception as e:
        yield None, f"❌ Lỗi xử lý Reference Audio: {str(e)}"
        return
    
    normalized_text = _text_normalizer.normalize(raw_text)
    text_chunks = split_text_into_chunks(normalized_text, max_chars=max_chars_chunk)
    total_chunks = len(text_chunks)
    
    if not text_chunks:
        yield None, "❌ Không có đoạn văn bản nào để tổng hợp."
        return
    
    # === STANDARD MODE ===
    if generation_mode == "Standard (Một lần)":
        # Note: use_batch and max_batch_size_run are available here but currently ignored/decoy
        yield None, f"🚀 Bắt đầu tổng hợp trên Intel XPU ({total_chunks} đoạn)..."
        
        all_wavs = []
        sr = 24000
        start_time = time.time()
        
        if use_batch and total_chunks > 1:
            try:
                for i in range(0, len(text_chunks), max_batch_size_run):
                    yield None, print(f" đang xử lý batch {i//max_batch_size_run + 1} ...")
                    batch_chunks = text_chunks[i : i + max_batch_size_run]
                    
                    # Gọi hàm infer_batch đã viết ở trên
                    batch_results = tts.infer_batch(
                        texts = batch_chunks,  
                        ref_codes=ref_codes, 
                        ref_text=ref_text_raw,
                        temperature=temperature
                    )
                    
                    if batch_results is not None and len(batch_results) > 0:
                        all_wavs.extend(batch_results)

                if not all_wavs:
                    yield None, "❌ Không sinh được audio nào."
                    return

                yield None, "💾 Đang ghép file và lưu..."
                
                final_wav = join_audio_chunks(all_wavs, sr=sr, silence_p=0.15)
            
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    sf.write(tmp.name, final_wav, sr)
                    output_path = tmp.name
                
                process_time = time.time() - start_time
                speed_info = f", Tốc độ: {len(final_wav)/sr/process_time:.2f}x realtime" if process_time > 0 else ""
            
                yield output_path, f"✅ Hoàn tất! (Thời gian: {process_time:.2f}s{speed_info}) (Backend: Intel XPU)"
                cleanup_gpu_memory()
                return
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                cleanup_gpu_memory()
                yield None, f"❌ Lỗi Standard Mode khi infer batch: {str(e)}"
                return
        try:
            # Sequential processing (Native XPU backend)
            for i, chunk in enumerate(text_chunks):
                yield None, f"⏳ Đang xử lý đoạn {i+1}/{total_chunks} (XPU Bfloat16)..."
                
                chunk_wav = tts.infer(
                    chunk, 
                    ref_codes=ref_codes, 
                    ref_text=ref_text_raw,
                    temperature=temperature,
                    skip_normalize=True
                )
                
                if chunk_wav is not None and len(chunk_wav) > 0:
                    all_wavs.append(chunk_wav)
            
            if not all_wavs:
                yield None, "❌ Không sinh được audio nào."
                return
            
            yield None, "💾 Đang ghép file và lưu..."
            
            final_wav = join_audio_chunks(all_wavs, sr=sr, silence_p=0.15)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                output_path = tmp.name
            
            process_time = time.time() - start_time
            speed_info = f", Tốc độ: {len(final_wav)/sr/process_time:.2f}x realtime" if process_time > 0 else ""
            
            yield output_path, f"✅ Hoàn tất! (Thời gian: {process_time:.2f}s{speed_info}) (Backend: Intel XPU)"
            cleanup_gpu_memory()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            cleanup_gpu_memory()
            yield None, f"❌ Lỗi Standard Mode: {str(e)}"
            return
    
    # === STREAMING MODE ===
    else:
        sr = 24000
        crossfade_samples = int(sr * 0.03)
        audio_queue = queue.Queue(maxsize=100)
        PRE_BUFFER_SIZE = 3
        
        end_event = threading.Event()
        error_event = threading.Event()
        error_msg = ""
        
        def producer_thread():
            nonlocal error_msg
            try:
                previous_tail = None
                
                for i, chunk_text in enumerate(text_chunks):
                    stream_gen = tts.infer_stream(
                        chunk_text, 
                        ref_codes=ref_codes, 
                        ref_text=ref_text_raw,
                        temperature=temperature,
                        skip_normalize=True
                    )
                    
                    for part_idx, audio_part in enumerate(stream_gen):
                        if audio_part is None or len(audio_part) == 0:
                            continue
                        
                        if previous_tail is not None and len(previous_tail) > 0:
                            overlap = min(len(previous_tail), len(audio_part), crossfade_samples)
                            if overlap > 0:
                                fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                                
                                blended = (audio_part[:overlap] * fade_in + 
                                         previous_tail[-overlap:] * fade_out)
                                
                                processed = np.concatenate([
                                    previous_tail[:-overlap] if len(previous_tail) > overlap else np.array([]),
                                    blended,
                                    audio_part[overlap:]
                                ])
                            else:
                                processed = np.concatenate([previous_tail, audio_part])
                            
                            tail_size = min(crossfade_samples, len(processed))
                            previous_tail = processed[-tail_size:].copy()
                            output_chunk = processed[:-tail_size] if len(processed) > tail_size else processed
                        else:
                            tail_size = min(crossfade_samples, len(audio_part))
                            previous_tail = audio_part[-tail_size:].copy()
                            output_chunk = audio_part[:-tail_size] if len(audio_part) > tail_size else audio_part
                        
                        if len(output_chunk) > 0:
                            audio_queue.put((sr, output_chunk))
                
                if previous_tail is not None and len(previous_tail) > 0:
                    audio_queue.put((sr, previous_tail))
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                error_event.set()
            finally:
                end_event.set()
                audio_queue.put(None)
        
        threading.Thread(target=producer_thread, daemon=True).start()
        
        yield (sr, np.zeros(int(sr * 0.05))), "📄 Đang buffering..."
        
        pre_buffer = []
        while len(pre_buffer) < PRE_BUFFER_SIZE:
            try:
                item = audio_queue.get(timeout=5.0)
                if item is None:
                    break
                pre_buffer.append(item)
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    return
                break
        
        full_audio_buffer = []
        for sr, audio_data in pre_buffer:
            full_audio_buffer.append(audio_data)
            yield (sr, audio_data), f"🔊 Đang phát (Intel XPU)..."
        
        while True:
            try:
                item = audio_queue.get(timeout=0.05)
                if item is None:
                    break
                sr, audio_data = item
                full_audio_buffer.append(audio_data)
                yield (sr, audio_data), f"🔊 Đang phát (Intel XPU)..."
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    break
                if end_event.is_set() and audio_queue.empty():
                    break
                continue
        
        if full_audio_buffer:
            final_wav = np.concatenate(full_audio_buffer)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                yield tmp.name, f"✅ Hoàn tất Streaming! (Intel XPU)"
            
            cleanup_gpu_memory()


# --- 4. UI SETUP ---
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui'],
).set(
    button_primary_background_fill="linear-gradient(90deg, #6366f1 0%, #0ea5e9 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4f46e5 0%, #0284c7 100%)",
)

css = """
.container { max-width: 1400px; margin: auto; }
.header-box {
    text-align: center;
    margin-bottom: 25px;
    padding: 25px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px;
    color: white !important;
}
.header-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: white !important;
}
.gradient-text {
    background: -webkit-linear-gradient(45deg, #60A5FA, #22D3EE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-icon {
    color: white;
}
.status-box {
    font-weight: 500;
    border: 1px solid rgba(99, 102, 241, 0.1);
    background: rgba(99, 102, 241, 0.03);
    border-radius: 8px;
}
.status-box textarea {
    text-align: center;
    font-family: inherit;
}
.model-card-content {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 15px;
    font-size: 0.9rem;
    text-align: center;
    color: white !important;
}
.model-card-item {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    color: white !important;
}
.model-card-item strong {
    color: white !important;
}
.model-card-item span {
    color: white !important;
}
.model-card-link {
    color: #60A5FA;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
}
.model-card-link:hover {
    color: #22D3EE;
    text-decoration: underline;
}
.warning-banner {
    background-color: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
}
.warning-banner-title {
    color: #0369a1;
    font-weight: 700;
    font-size: 1.1rem;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}
.warning-banner-content {
    color: #0c4a6e;
    font-size: 0.9rem;
    line-height: 1.5;
}
"""

head_html = """
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🦜</text></svg>">
"""

with gr.Blocks(theme=theme, css=css, title="VieNeu-TTS (XPU)", head=head_html) as demo:

    with gr.Column(elem_classes="container"):
        gr.HTML("""
<div class="header-box">
    <h1 class="header-title">
        <span class="header-icon">🦜</span>
        <span class="gradient-text">VieNeu-TTS Studio (Intel XPU Edition)</span>
    </h1>
    <div class="model-card-content">
        <div class="model-card-item">
            <strong>Models:</strong>
            <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS" target="_blank" class="model-card-link">VieNeu-TTS</a>
            <span>•</span>
            <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B" target="_blank" class="model-card-link">VieNeu-TTS-0.3B</a>
        </div>
        <div class="model-card-item">
            <strong>Repository:</strong>
            <a href="https://github.com/pnnbao97/VieNeu-TTS" target="_blank" class="model-card-link">GitHub</a>
        </div>
        <div class="model-card-item">
            <strong>Tác giả:</strong>
            <a href="https://www.facebook.com/pnnbao97" target="_blank" class="model-card-link">Phạm Nguyễn Ngọc Bảo</a>
        </div>
        <div class="model-card-item">
            <strong>Discord:</strong>
            <a href="https://discord.gg/yJt8kzjzWZ" target="_blank" class="model-card-link">Tham gia cộng đồng</a>
        </div>
    </div>
</div>
        """)
        
        # --- CONFIGURATION ---
        with gr.Group():
            with gr.Row():
                backbone_select = gr.Dropdown(
                    list(BACKBONE_CONFIGS.keys()) + ["Custom Model"], 
                    value="VieNeu-TTS (GPU)", 
                    label="🦜 Backbone"
                )
                codec_select = gr.Dropdown(list(CODEC_CONFIGS.keys()), value="NeuCodec (Distill)", label="🎵 Codec")
                device_choice = gr.Radio(get_available_devices(), value="XPU", label="🖥️ Device", interactive=False)
            
            with gr.Row(visible=False) as custom_model_group:
                custom_backbone_model_id = gr.Textbox(
                    label="📦 Custom Model ID",
                    placeholder="pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen",
                    info="Nhập HuggingFace Repo ID hoặc đường dẫn local",
                    scale=2
                )
                custom_backbone_hf_token = gr.Textbox(
                    label="🔑 HF Token (nếu private)",
                    placeholder="Để trống nếu repo public",
                    type="password",
                    info="Token để truy cập repo private",
                    scale=1
                )
                custom_backbone_base_model = gr.Dropdown(
                    [k for k in BACKBONE_CONFIGS.keys() if "gguf" not in k.lower()],
                    label="🔗 Base Model (cho LoRA)",
                    value="VieNeu-TTS-0.3B (GPU)",
                    visible=False,
                    info="Model gốc để merge với LoRA",
                    scale=1
                )
            
            gr.Markdown("""
            💡 **Sử dụng Custom Model:** Chọn "Custom Model" để tải LoRA adapter hoặc bất kỳ model nào được finetune từ **VieNeu-TTS** hoặc **VieNeu-TTS-0.3B**.
            """)
            
            gr.HTML("""
            <div class="warning-banner">
                <div class="warning-banner-title">
                    ⚡ Chế độ tối ưu hóa cho Intel Arc GPU (XPU)
                </div>
                <div class="warning-banner-content">
                    Ứng dụng đang chạy trên pytorch nightly tối ưu hóa riêng cho card đồ họa Intel (PyTorch XPU).<br>
                    Lần tạo giọng nói <b>đầu tiên</b> (hoặc sau khi tải model mới) sẽ lâu hơn 1 chút. Các lần tiếp theo tốc độ sẽ được tối ưu.
                </div>
            </div>
            """)

            btn_load = gr.Button("🔄 Tải Model", variant="primary")
            model_status = gr.Markdown("⏳ Chưa tải model.")
        
        with gr.Row(elem_classes="container"):
            # --- INPUT ---
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label=f"Văn bản",
                    lines=4,
                    value="Hà Nội, trái tim của Việt Nam, là một thành phố ngàn năm văn hiến với bề dày lịch sử và văn hóa độc đáo. Bước chân trên những con phố cổ kính quanh Hồ Hoàn Kiếm, du khách như được du hành ngược thời gian, chiêm ngưỡng kiến trúc Pháp cổ điển hòa quyện với nét kiến trúc truyền thống Việt Nam. Mỗi con phố trong khu phố cổ mang một tên gọi đặc trưng, phản ánh nghề thủ công truyền thống từng thịnh hành nơi đây như phố Hàng Bạc, Hàng Đào, Hàng Mã. Ẩm thực Hà Nội cũng là một điểm nhấn đặc biệt, từ tô phở nóng hổi buổi sáng, bún chả thơm lừng trưa hè, đến chè Thái ngọt ngào chiều thu. Những món ăn dân dã này đã trở thành biểu tượng của văn hóa ẩm thực Việt, được cả thế giới yêu mến. Người Hà Nội nổi tiếng với tính cách hiền hòa, lịch thiệp nhưng cũng rất cầu toàn trong từng chi tiết nhỏ, từ cách pha trà sen cho đến cách chọn hoa sen tây để thưởng trà.",
                )
                
                with gr.Tabs() as tabs:
                    with gr.TabItem("👤 Preset", id="preset_mode") as tab_preset:
                        voice_select = gr.Dropdown(choices=[], value=None, label="Giọng mẫu")
                    
                    with gr.TabItem("🦜 Voice Cloning", id="custom_mode") as tab_custom:
                        custom_audio = gr.Audio(label="Audio giọng mẫu (3-5 giây) (.wav)", type="filepath")
                        custom_text = gr.Textbox(label="Nội dung audio mẫu - vui lòng gõ đúng nội dung của audio mẫu - kể cả dấu câu vì model rất nhạy cảm với dấu câu (.,?!)")
                        gr.Examples(
                            examples=[
                                [os.path.join("examples", "audio_ref", "example.wav"), "Ví dụ 2. Tính trung bình của dãy số."],
                                [os.path.join("examples", "audio_ref", "example_2.wav"), "Trên thực tế, các nghi ngờ đã bắt đầu xuất hiện."],
                                [os.path.join("examples", "audio_ref", "example_3.wav"), "Cậu có nhìn thấy không?"],
                                [os.path.join("examples", "audio_ref", "example_4.wav"), "Tết là dịp mọi người háo hức đón chào một năm mới với nhiều hy vọng và mong ước."]
                            ],
                            inputs=[custom_audio, custom_text],
                            label="Ví dụ mẫu để thử nghiệm clone giọng"
                        )
                        
                        gr.Markdown("""
                        **💡 Mẹo nhỏ:** Nếu kết quả Zero-shot Voice Cloning chưa như ý, bạn hãy cân nhắc **Finetune (LoRA)** để đạt chất lượng tốt nhất. 
                        Hướng dẫn chi tiết có tại file: `finetune/README.md` hoặc xem trên [GitHub](https://github.com/pnnbao97/VieNeu-TTS/tree/main/finetune).
                        """)              
                
                generation_mode = gr.Radio(
                    ["Standard (Một lần)"],
                    value="Standard (Một lần)",
                    label="Chế độ sinh"
                )

                with gr.Row():
                    use_batch = gr.Checkbox(
                        value=True, 
                        label="⚡ Batch Processing",
                        info="Xử lý nhiều đoạn cùng lúc. Nên luôn chọn bật để tăng tốc độ."
                    )
                    max_batch_size_run = gr.Slider(
                        minimum=1, 
                        maximum=256, 
                        value=128, 
                        step=1, 
                        label="📊 Batch Size (Generation)",
                        info="Số lượng đoạn văn bản xử lý cùng lúc. Càng lớn thì xử lý càng nhanh. Thông thường 128 chunks với 64 chars hết 7gb vram."
                    )
                
                with gr.Accordion("⚙️ Cài đặt nâng cao (Generation)", open=False):
                    with gr.Row():
                        temperature_slider = gr.Slider(
                            minimum=0.1, maximum=1.5, value=1.0, step=0.1,
                            label="🌡️ Temperature", 
                            info="Độ sáng tạo. Cao = đa dạng cảm xúc hơn nhưng dễ lỗi. Thấp = ổn định hơn."
                        )
                        max_chars_chunk_slider = gr.Slider(
                            minimum=64, maximum=512, value=128, step=16,
                            label="📝 Max Chars per Chunk",
                            info="Độ dài tối đa mỗi đoạn xử lý. Càng nhỏ thì xử lý càng nhanh nếu tăng số Batch Size lên."
                        )
                
                current_mode_state = gr.State("preset_mode")
                
                with gr.Row():
                    btn_generate = gr.Button("🎵 Bắt đầu", variant="primary", scale=2, interactive=False)
                    btn_stop = gr.Button("⏹️ Dừng", variant="stop", scale=1, interactive=False)
            
            # --- OUTPUT ---
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Kết quả",
                    type="filepath",
                    autoplay=True
                )
                status_output = gr.Textbox(
                    label="Trạng thái", 
                    elem_classes="status-box",
                    lines=2,
                    max_lines=10,
                    show_copy_button=True
                )
                gr.Markdown("<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>🔒 Audio được đóng dấu bản quyền ẩn (Watermarker) để bảo mật và định danh AI.</div>")
        
        # --- EVENT HANDLERS ---
        def on_codec_change(codec: str, current_mode: str):
            is_onnx = "onnx" in codec.lower()
            if is_onnx and current_mode == "custom_mode":
                return gr.update(visible=False), gr.update(selected="preset_mode"), "preset_mode"
            return gr.update(visible=not is_onnx), gr.update(), current_mode
        
        codec_select.change(
            on_codec_change, 
            inputs=[codec_select, current_mode_state], 
            outputs=[tab_custom, tabs, current_mode_state]
        )
        
        tab_preset.select(lambda: "preset_mode", outputs=current_mode_state)
        tab_custom.select(lambda: "custom_mode", outputs=current_mode_state)
        
        def on_backbone_change(choice):
            is_custom = (choice == "Custom Model")
            return gr.update(visible=is_custom)

        backbone_select.change(
            on_backbone_change,
            inputs=[backbone_select],
            outputs=[custom_model_group]
        )
        
        def on_custom_id_change(model_id):
            if model_id and "lora" in model_id.lower():
                if "0.3" in model_id:
                    base_model = "VieNeu-TTS-0.3B (GPU)"
                else:
                    base_model = "VieNeu-TTS (GPU)"
                return (gr.update(visible=True, value=base_model), gr.update(), gr.update())
            return (gr.update(visible=False), gr.update(), gr.update())
            
        custom_backbone_model_id.change(
            on_custom_id_change,
            inputs=[custom_backbone_model_id],
            outputs=[custom_backbone_base_model, custom_audio, custom_text]
        )

        btn_load.click(
            fn=load_model,
            inputs=[backbone_select, codec_select, device_choice, 
                    custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
            outputs=[model_status, btn_generate, btn_load, btn_stop, voice_select, tab_preset, tab_custom, tabs, current_mode_state]
        )
        
        generate_event = btn_generate.click(
            fn=synthesize_speech,
            inputs=[text_input, voice_select, custom_audio, custom_text, current_mode_state, 
                    generation_mode, use_batch, max_batch_size_run,
                    temperature_slider, max_chars_chunk_slider],
            outputs=[audio_output, status_output]
        )
        
        btn_generate.click(lambda: gr.update(interactive=True), outputs=btn_stop)
        generate_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)
        
        btn_stop.click(fn=None, cancels=[generate_event])
        btn_stop.click(lambda: (None, "⏹️ Đã dừng tạo giọng nói."), outputs=[audio_output, status_output])
        btn_stop.click(lambda: gr.update(interactive=False), outputs=btn_stop)

        demo.load(
            fn=restore_ui_state,
            outputs=[model_status, btn_generate, btn_stop]
        )

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    is_on_colab = os.getenv("COLAB_RELEASE_TAG") is not None
    share = env_bool("GRADIO_SHARE", default=is_on_colab)

    if server_name == "0.0.0.0" and os.getenv("GRADIO_SHARE") is None:
        share = False

    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)
