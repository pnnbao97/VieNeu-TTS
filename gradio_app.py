import gradio as gr
print("⏳ Đang khởi động VieNeu-TTS... Vui lòng chờ...")
import soundfile as sf
import tempfile
import torch
from vieneu import VieNeuTTS, FastVieNeuTTS
import os
import sys
import time
import numpy as np
from typing import Generator, Optional, Tuple
import queue
import threading
import yaml
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks, env_bool
from vieneu_utils.normalize_text import VietnameseTTSNormalizer
from functools import lru_cache
import gc

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
MAX_TOTAL_CHARS_STREAMING = _text_settings.get("max_total_chars_streaming", 3000)

if not BACKBONE_CONFIGS or not CODEC_CONFIGS:
    raise ValueError("config.yaml thiếu backbone_configs hoặc codec_configs")

# --- 1. MODEL CONFIGURATION ---
# Global model instance
tts = None
current_backbone = None
current_codec = None
model_loaded = False
using_lmdeploy = False

# Normalizer (module-level singleton)
_text_normalizer = VietnameseTTSNormalizer()

# Cache for reference texts
_ref_text_cache = {}

def get_available_devices() -> list[str]:
    """Get list of available devices for current platform."""
    devices = ["Auto", "CPU"]

    if sys.platform == "darwin":
        # macOS - check MPS
        if torch.backends.mps.is_available():
            devices.append("MPS")
    else:
        # Windows/Linux - check CUDA
        if torch.cuda.is_available():
            devices.append("CUDA")

    return devices

def get_model_status_message() -> str:
    """Reconstruct status message from global state"""
    global model_loaded, tts, using_lmdeploy, current_backbone, current_codec
    if not model_loaded or tts is None:
        return "⏳ Chưa tải model."
    
    backbone_config = BACKBONE_CONFIGS.get(current_backbone, {})
    codec_config = CODEC_CONFIGS.get(current_codec, {})
    
    backend_name = "🚀 LMDeploy (Optimized)" if using_lmdeploy else "📦 Standard"
    
    # We don't track the exact device strings perfectly in global state, so we estimate
    device_info = "GPU" if using_lmdeploy else "Auto"
    codec_device = "CPU" if "ONNX" in (current_codec or "") else ("GPU/MPS" if torch.cuda.is_available() or torch.backends.mps.is_available() else "CPU")
    
    preencoded_note = "\n⚠️ Codec ONNX không hỗ trợ chức năng clone giọng nói." if codec_config.get('use_preencoded') else ""
    
    opt_info = ""
    if using_lmdeploy and hasattr(tts, 'get_optimization_stats'):
        stats = tts.get_optimization_stats()
        opt_info = (
            f"\n\n🔧 Tối ưu hóa:"
            f"\n  • Triton: {'✅' if stats['triton_enabled'] else '❌'}"
            f"\n  • Max Batch Size (Default): {stats.get('max_batch_size', 'N/A')}"
            f"\n  • Reference Cache: {stats['cached_references']} voices"
            f"\n  • Prefix Caching: ✅"
        )

    return (
        f"✅ Model đã tải thành công!\n\n"
        f"🔧 Backend: {backend_name}\n"
        f"🦜 Backbone: {current_backbone}\n"
        f"🎵 Codec: {current_codec}{preencoded_note}{opt_info}"
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

def should_use_lmdeploy(backbone_choice: str, device_choice: str) -> bool:
    """Determine if we should use LMDeploy backend."""
    # LMDeploy not supported on macOS
    if sys.platform == "darwin":
        return False

    if "gguf" in backbone_choice.lower():
        return False

    if device_choice == "Auto":
        has_gpu = torch.cuda.is_available()
    elif device_choice == "CUDA":
        has_gpu = torch.cuda.is_available()
    else:
        has_gpu = False

    return has_gpu

@lru_cache(maxsize=32)
def get_ref_text_cached(text_path: str) -> str:
    """Cache reference text loading"""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()

def cleanup_gpu_memory():
    """Aggressively cleanup GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def load_model(backbone_choice: str, codec_choice: str, device_choice: str, 
               force_lmdeploy: bool, custom_model_id: str = "", custom_base_model: str = "", 
               custom_hf_token: str = ""):
    """Load model with optimizations and max batch size control"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    lmdeploy_error_reason = None
    model_loaded = False # Ensure we don't try to use a half-loaded model
    
    yield (
        "⏳ Đang tải model với tối ưu hóa... Lưu ý: Quá trình này sẽ tốn thời gian. Vui lòng kiên nhẫn.",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update()
    )
    
    try:
        # Cleanup before loading new model
        if tts is not None:
            tts = None # Reset instead of del to avoid NameError if load fails
            cleanup_gpu_memory()
        
        # Prepare Backbone Config/Repo
        custom_loading = False
        is_merged_lora = False

        if backbone_choice == "Custom Model":
            custom_loading = True
            if not custom_model_id or not custom_model_id.strip():
                yield (
                    "❌ Lỗi: Vui lòng nhập Model ID cho Custom Model.",
                    gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update()
                )
                return

            # Check if it is a LoRA to merge
            if "lora" in custom_model_id.lower():
                # Merging mode
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
                    "repo": base_config["repo"], # Load base first
                    "supports_streaming": base_config["supports_streaming"],
                    "description": f"Custom Merged: {custom_model_id} + {custom_base_model}"
                }
                is_merged_lora = True
            else:
                # Normal custom model
                backbone_config = {
                    "repo": custom_model_id.strip(),
                    "supports_streaming": False, # Assume false for unknown
                    "description": f"Custom Model: {custom_model_id}"
                }
        else:
            backbone_config = BACKBONE_CONFIGS[backbone_choice]
            
        codec_config = CODEC_CONFIGS[codec_choice]
        
        # Override LMDeploy if custom
        if custom_loading:
             if "gguf" in backbone_config['repo'].lower():
                 # GGUF must use Standard backend (llama-cpp)
                 use_lmdeploy = False
             elif is_merged_lora:
                 # LoRA can use LMDeploy if we merge first (checked logic below) or Standard
                 use_lmdeploy = force_lmdeploy and should_use_lmdeploy(custom_base_model, device_choice)
             else:
                 # Full custom model (e.g. finetune)
                 use_lmdeploy = force_lmdeploy and should_use_lmdeploy("VieNeu-TTS (GPU)", device_choice) # Assume GPU compatible?
        else:
             use_lmdeploy = force_lmdeploy and should_use_lmdeploy(backbone_choice, device_choice)
        
        if use_lmdeploy:
            lmdeploy_error_reason = None
            print(f"🚀 Using LMDeploy backend with optimizations")
            
            backbone_device = "cuda"
            
            if "ONNX" in codec_choice:
                codec_device = "cpu"
            else:
                codec_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Special handling for Custom LoRA + LMDeploy -> Merge & Save
            target_backbone_repo = backbone_config["repo"]
            
            if custom_loading and is_merged_lora:
                safe_name = custom_model_id.strip().replace("/", "_").replace("\\", "_").replace(":", "")
                cache_dir = os.path.join("merged_models_cache", safe_name)
                target_backbone_repo = os.path.abspath(cache_dir)
                
                # Check if already merged (and voices.json exists)
                if not os.path.exists(cache_dir) or not os.path.exists(os.path.join(cache_dir, "vocab.json")):
                    print(f"🔄 Merging LoRA for LMDeploy optimization: {cache_dir}")
                    if os.path.exists(cache_dir):
                        print("   ⚠️ Detected incomplete cache, rebuilding...")
                    yield (
                         f"⏳ Đang merge và lưu model LoRA để tối ưu cho LMDeploy (thao tác này chỉ chạy một lần)...",
                         gr.update(interactive=False),
                         gr.update(interactive=False),
                         gr.update(interactive=False),
                         gr.update(),
                         gr.update(), gr.update(), gr.update(), gr.update()
                    )
                    
                    try:
                        # Use GPU for merging if available for speed
                        # We use the Base Model specified
                        base_repo = BACKBONE_CONFIGS[custom_base_model]["repo"]
                        merge_device = "cuda" if torch.cuda.is_available() else "cpu"
                        
                        print(f"   • Loading base: {base_repo} ({merge_device})")
                        temp_tts = VieNeuTTS(
                            backbone_repo=base_repo,
                            backbone_device=merge_device, 
                            codec_repo=codec_config["repo"],
                            codec_device="cpu", # Codec unused for merging, keep on CPU
                            hf_token=custom_hf_token
                        )
                        
                        print(f"   • Loading Adapter: {custom_model_id}")
                        temp_tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                        
                        print(f"   • Merging...")
                        if hasattr(temp_tts.backbone, "merge_and_unload"):
                            temp_tts.backbone = temp_tts.backbone.merge_and_unload()
                        
                        print(f"   • Saving to cache: {cache_dir}")
                        temp_tts.backbone.save_pretrained(cache_dir)
                        temp_tts.tokenizer.save_pretrained(cache_dir)
                        
                        # Fix for LMDeploy: Explicitly save legacy tokenizer files (vocab.json, merges.txt)
                        # because LMDeploy/Transformers might default to slow tokenizer if fast one has issues,
                        # and save_pretrained on fast tokenizer sometimes omits legacy files.
                        try:
                            print("   • Ensuring legacy tokenizer files...")
                            from transformers import AutoTokenizer
                            slow_tokenizer = AutoTokenizer.from_pretrained(base_repo, use_fast=False)
                            slow_tokenizer.save_pretrained(cache_dir)
                        except Exception as e:
                            print(f"   ⚠️ Warning: Could not save slow tokenizer files: {e}")

                        # Save voices.json to cache directory so FastVieNeuTTS can find it
                        print(f"   • Saving voices definition...")
                        import json
                        voices_json_path = os.path.join(cache_dir, "voices.json")
                        voices_content = {
                             "meta": { "note": "Automatically generated during LoRA merge" },
                             "default_voice": temp_tts._default_voice,
                             "presets": temp_tts._preset_voices
                        }
                        with open(voices_json_path, 'w', encoding='utf-8') as f:
                             json.dump(voices_content, f, ensure_ascii=False, indent=2)

                        del temp_tts
                        cleanup_gpu_memory()
                        print("   ✅ Merge & Save successfully!")
                        
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise RuntimeError(f"Failed to merge & save LoRA for LMDeploy: {e}")

            print(f"📦 Loading optimized model...")
            print(f"   Backbone: {target_backbone_repo} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")
            print(f"   Triton: Enabled")
            
            try:
                tts = FastVieNeuTTS(
                    backbone_repo=target_backbone_repo,
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    memory_util=0.3,
                    tp=1,
                    enable_prefix_caching=True,
                    enable_triton=True,
                    hf_token=custom_hf_token
                )
                using_lmdeploy = True
                
                # Legacy caching removed
                print(f"   ✅ Optimized backend initialized")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                
                error_str = str(e)
                if "$env:CUDA_PATH" in error_str:
                    lmdeploy_error_reason = "Không tìm thấy biến môi trường CUDA_PATH. Vui lòng cài đặt NVIDIA GPU Computing Toolkit."
                else:
                    lmdeploy_error_reason = f"{error_str}"
                
                yield (
                    f"⚠️ LMDeploy Init Error: {lmdeploy_error_reason}. Đang loading model với backend mặc định - tốc độ chậm hơn so với lmdeploy...",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update()
                )
                time.sleep(1)
                use_lmdeploy = False
                using_lmdeploy = False
        
        if not use_lmdeploy:
            print(f"📦 Using original backend")

            if device_choice == "Auto":
                if "gguf" in backbone_config['repo'].lower():
                    # GGUF: uses Metal on Mac, CUDA on Windows/Linux
                    if sys.platform == "darwin":
                        backbone_device = "gpu"  # llama-cpp-python uses Metal
                    else:
                        backbone_device = "gpu" if torch.cuda.is_available() else "cpu"
                else:
                    # PyTorch model
                    if sys.platform == "darwin":
                        backbone_device = "mps" if torch.backends.mps.is_available() else "cpu"
                    else:
                        backbone_device = "cuda" if torch.cuda.is_available() else "cpu"

                # Codec device
                if "ONNX" in codec_choice:
                    codec_device = "cpu"
                elif sys.platform == "darwin":
                    codec_device = "mps" if torch.backends.mps.is_available() else "cpu"
                else:
                    codec_device = "cuda" if torch.cuda.is_available() else "cpu"

            elif device_choice == "MPS":
                backbone_device = "mps"
                codec_device = "mps" if "ONNX" not in codec_choice else "cpu"

            else:
                backbone_device = device_choice.lower()
                codec_device = device_choice.lower()

                if "ONNX" in codec_choice:
                    codec_device = "cpu"

            if "gguf" in backbone_config['repo'].lower() and backbone_device == "cuda":
                backbone_device = "gpu"
            
            print(f"📦 Loading model...")
            print(f"   Backbone: {backbone_config['repo']} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")
            
            tts = VieNeuTTS(
                backbone_repo=backbone_config["repo"],
                backbone_device=backbone_device,
                codec_repo=codec_config["repo"],
                codec_device=codec_device,
                hf_token=custom_hf_token
            )

            # Perform LoRA Merge if needed (ONLY for Standard Backend)
            # For LMDeploy, we handled it above by saving to disk
            if is_merged_lora and custom_loading and not using_lmdeploy:
                yield (
                    f"🔄 Đang tải và merge LoRA adapter: {custom_model_id}...",
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update()
                )
                try:
                    # 1. Load Adapter
                    tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                    
                    # 2. Merge and Unload
                    # Check if backbone matches expected type for merge
                    if hasattr(tts, 'backbone') and hasattr(tts.backbone, 'merge_and_unload'):
                        print("   🔄 Merging LoRA into backbone...")
                        tts.backbone = tts.backbone.merge_and_unload()
                        
                        # Reset LoRA state so it behaves like a normal model
                        tts._lora_loaded = False 
                        tts._current_lora_repo = None
                        print("   ✅ Merged successfully!")
                    else:
                        print("   ⚠️ Warning: Model does not support merge_and_unload, keeping adapter active.")
                        
                except Exception as e:
                     raise RuntimeError(f"Failed to merge LoRA: {e}")

            using_lmdeploy = False
        
        current_backbone = backbone_choice
        current_codec = codec_choice
        model_loaded = True
        
        # Success message with optimization info
        backend_name = "🚀 LMDeploy (Optimized)" if using_lmdeploy else "📦 Standard"
        device_info = "cuda" if use_lmdeploy else (backbone_device if not use_lmdeploy else "N/A")
        
        streaming_support = "✅ Có" if backbone_config['supports_streaming'] else "❌ Không"
        preencoded_note = "\n⚠️ Codec này cần sử dụng pre-encoded codes (.pt files)" if codec_config['use_preencoded'] else ""
        
        opt_info = ""
        if using_lmdeploy and hasattr(tts, 'get_optimization_stats'):
            stats = tts.get_optimization_stats()
            opt_info = (
                f"\n\n🔧 Tối ưu hóa:"
                f"\n  • Triton: {'✅' if stats['triton_enabled'] else '❌'}"
                f"\n  • Max Batch Size (Default): {stats.get('max_batch_size', 'N/A')}"
                f"\n  • Reference Cache: {stats['cached_references']} voices"
                f"\n  • Prefix Caching: ✅"
            )
        
        warning_msg = ""
        if lmdeploy_error_reason:
             warning_msg = (
                 f"\n\n⚠️ **Cảnh báo:** Không thể kích hoạt LMDeploy (Optimized Backend) do lỗi sau:\n"
                 f"👉 {lmdeploy_error_reason}\n"
                 f"💡 Hệ thống đã tự động chuyển về chế độ Standard (chậm hơn)."
             )

        success_msg = get_model_status_message()
        if warning_msg:
            success_msg += warning_msg
            
        # Prepare voice update
        try:
            # Get voices with descriptions for UI from SDK
            voices = tts.list_preset_voices()
        except Exception:
            voices = []

        has_voices = len(voices) > 0
        
        if has_voices:
            default_v = tts._default_voice
            
            # Helper to get values list
            is_tuple = (len(voices) > 0 and isinstance(voices[0], tuple))
            voice_values = [v[1] for v in voices] if is_tuple else voices
            
            if not default_v and voice_values:
                 default_v = voice_values[0]

            # Ensure default_v is in the list and selected correctly
            if default_v and default_v not in voice_values:
                if is_tuple:
                    # Try to find a nice description if possible, else use ID
                    voices.append((default_v, default_v))
                else:
                    voices.append(default_v)
            
            # Sort voices by name/label for better UX
            if is_tuple:
                voices.sort(key=lambda x: str(x[0]))
            else:
                voices.sort()

            voice_update = gr.update(choices=voices, value=default_v, interactive=True)
            
            # Show Standard Tabs
            tab_p = gr.update(visible=True)
            tab_c = gr.update(visible=True)
            tab_sel = gr.update(selected="preset_mode")
            mode_state = "preset_mode"
        else:
            # Missing voices.json case
            msg = "⚠️ Không tìm thấy file voices.json. Vui lòng dùng Tab Voice Cloning."
            voice_update = gr.update(choices=[msg], value=msg, interactive=False)
            
            # Show Preset Tab (to see message) and Custom Tab
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
        using_lmdeploy = False

        if "$env:CUDA_PATH" in str(e):
            yield (
                "❌ Lỗi khi tải model: Không tìm thấy biến môi trường CUDA_PATH. Vui lòng cài đặt NVIDIA GPU Computing Toolkit (https://developer.nvidia.com/cuda/toolkit)",
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update()
            )
        else: 
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
                      mode_tab: str, generation_mode: str, use_batch: bool, max_batch_size_run: int,
                      temperature: float, max_chars_chunk: int):
    """Synthesis with optimization support and max batch size control"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    
    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return
    
    if not text or text.strip() == "":
        yield None, "⚠️ Vui lòng nhập văn bản!"
        return
    
    raw_text = text.strip()
    
    codec_config = CODEC_CONFIGS[current_codec]
    use_preencoded = codec_config['use_preencoded']
    
    
    # Setup Reference
    yield None, "📄 Đang xử lý Reference..."
    
    try:
        ref_codes = None
        ref_text_raw = ""
        
        if mode_tab == "preset_mode":
            if not voice_choice:
                raise ValueError("Vui lòng chọn giọng mẫu.")
            if "⚠️" in voice_choice:
                raise ValueError("Không có giọng mẫu khả dụng. Vui lòng chuyển sang Tab Voice Cloning.")
            
            # Use SDK method - handles caching and JSON internally
            voice_data = tts.get_preset_voice(voice_choice)
            ref_codes = voice_data['codes']
            ref_text_raw = voice_data['text']
            
        elif mode_tab == "custom_mode":
            # Reference from Custom Cloning UI
            if custom_audio is None:
                 raise ValueError("Vui lòng upload file Audio mẫu (Reference Audio)!")
            if not custom_text or not custom_text.strip():
                 raise ValueError("Vui lòng nhập nội dung văn bản của Audio mẫu (Reference Text)!")
            
            ref_text_raw = custom_text.strip()
            ref_codes = tts.encode_reference(custom_audio)
            
        else:
            raise ValueError(f"Unknown mode: {mode_tab}")

        # Ensure numpy for inference
        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()

    except Exception as e:
        yield None, f"❌ Lỗi xử lý Reference Audio: {str(e)}"
        return
    
    # === STANDARD MODE ===
    if generation_mode == "Standard (Một lần)":
        backend_name = "LMDeploy" if using_lmdeploy else "Standard"

        normalized_text = _text_normalizer.normalize(raw_text)
        text_chunks = split_text_into_chunks(normalized_text, max_chars=max_chars_chunk)
        total_chunks = len(text_chunks)

        batch_info = " (Batch Mode)" if use_batch and using_lmdeploy and total_chunks > 1 else ""
        
        # Show batch size info
        batch_size_info = ""
        if use_batch and using_lmdeploy and hasattr(tts, 'max_batch_size'):
            batch_size_info = f" [Max batch: {tts.max_batch_size}]"
        
        yield None, f"🚀 Bắt đầu tổng hợp {backend_name}{batch_info}{batch_size_info} ({total_chunks} đoạn)..."
        
        all_wavs = []
        sr = 24000
        
        start_time = time.time()
        
        try:
            # Use batch processing if enabled and using LMDeploy
            if use_batch and using_lmdeploy and hasattr(tts, 'infer_batch') and total_chunks > 1:
                # Show how many mini-batches will be processed
                num_batches = (total_chunks + max_batch_size_run - 1) // max_batch_size_run
                
                yield None, f"⚡ Xử lý {num_batches} mini-batch(es) (max {max_batch_size_run} đoạn/batch)..."
                
                chunk_wavs = tts.infer_batch(
                    text_chunks, 
                    ref_codes=ref_codes, 
                    ref_text=ref_text_raw,
                    max_batch_size=max_batch_size_run,
                    temperature=temperature
                )
                
                for chunk_wav in chunk_wavs:
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_wavs.append(chunk_wav)

            else:
                # Sequential processing
                for i, chunk in enumerate(text_chunks):
                    yield None, f"⏳ Đang xử lý đoạn {i+1}/{total_chunks}..."
                    
                    chunk_wav = tts.infer(
                        chunk, 
                        ref_codes=ref_codes, 
                        ref_text=ref_text_raw,
                        temperature=temperature,
                        max_chars=max_chars_chunk,
                        skip_normalize=True
                    )
                    
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_wavs.append(chunk_wav)
            
            if not all_wavs:
                yield None, "❌ Không sinh được audio nào."
                return
            
            yield None, "💾 Đang ghép file và lưu..."
            
            # Use utility function for joining with silence/crossfade
            # Default silence=0.15s to match SDK
            final_wav = join_audio_chunks(all_wavs, sr=sr, silence_p=0.15)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                output_path = tmp.name
            
            process_time = time.time() - start_time
            backend_info = f" (Backend: {'LMDeploy 🚀' if using_lmdeploy else 'Standard 📦'})"
            speed_info = f", Tốc độ: {len(final_wav)/sr/process_time:.2f}x realtime" if process_time > 0 else ""
            
            
            yield output_path, f"✅ Hoàn tất! (Thời gian: {process_time:.2f}s{speed_info}){backend_info}"
            
            # Cleanup memory
            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            
            cleanup_gpu_memory()
            
        except torch.cuda.OutOfMemoryError as e:
            cleanup_gpu_memory()
            yield None, (
                f"❌ GPU hết VRAM! Hãy thử:\n"
                f"• Giảm Max Batch Size (hiện tại: {tts.max_batch_size if hasattr(tts, 'max_batch_size') else 'N/A'})\n"
                f"• Giảm độ dài văn bản\n\n"
                f"Chi tiết: {str(e)}"
            )
            return
            
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
        
        normalized_text = _text_normalizer.normalize(raw_text)
        text_chunks = split_text_into_chunks(normalized_text, max_chars=max_chars_chunk)
        
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
                        max_chars=max_chars_chunk,
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
        backend_info = "🚀 LMDeploy" if using_lmdeploy else "📦 Standard"
        for sr, audio_data in pre_buffer:
            full_audio_buffer.append(audio_data)
            yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
        
        while True:
            try:
                item = audio_queue.get(timeout=0.05)
                if item is None:
                    break
                sr, audio_data = item
                full_audio_buffer.append(audio_data)
                yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
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
                
                yield tmp.name, f"✅ Hoàn tất Streaming! ({backend_info})"
            
            # Cleanup memory
            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            
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
    background-color: #fffbeb;
    border: 1px solid #fef3c7;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
}
.warning-banner-title {
    color: #92400e;
    font-weight: 700;
    font-size: 1.1rem;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}
.warning-banner-grid {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
}
.warning-banner-item {
    flex: 1;
    min-width: 240px;
    background: #fef3c7;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #fde68a;
}
.warning-banner-item strong {
    color: #b45309;
    display: block;
    margin-bottom: 4px;
    font-size: 0.95rem;
}
.warning-banner-content {
    color: #78350f;
    font-size: 0.9rem;
    line-height: 1.5;
}
.warning-banner-content b {
    color: #451a03;
    background: rgba(251, 191, 36, 0.2);
    padding: 1px 4px;
    border-radius: 4px;
}
"""

EXAMPLES_LIST = [
    ["Về miền Tây không chỉ để ngắm nhìn sông nước hữu tình, mà còn để cảm nhận tấm chân tình của người dân nơi đây.", "Vĩnh (nam miền Nam)"],
    ["Hà Nội những ngày vào thu mang một vẻ đẹp trầm mặc và cổ kính đến lạ thường.", "Bình (nam miền Bắc)"],
]


# Favicon (Parrot Emoji)
head_html = """
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🦜</text></svg>">
"""

with gr.Blocks(theme=theme, css=css, title="VieNeu-TTS", head=head_html) as demo:

    with gr.Column(elem_classes="container"):
        gr.HTML("""
<div class="header-box">
    <h1 class="header-title">
        <span class="header-icon">🦜</span>
        <span class="gradient-text">VieNeu-TTS Studio</span>
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
                device_choice = gr.Radio(get_available_devices(), value="Auto", label="🖥️ Device")
            
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
            
            with gr.Row():
                use_lmdeploy_cb = gr.Checkbox(
                    value=True, 
                    label="🚀 Optimize with LMDeploy (Khuyên dùng cho NVIDIA GPU)",
                    info="Tick nếu bạn dùng GPU để tăng tốc độ tổng hợp đáng kể."
                )
            
            
            gr.Markdown("""
            💡 **Sử dụng Custom Model:** Chọn "Custom Model" để tải LoRA adapter hoặc bất kỳ model nào được finetune từ **VieNeu-TTS** hoặc **VieNeu-TTS-0.3B**.
            """)
            
            gr.HTML("""
            <div class="warning-banner">
                <div class="warning-banner-title">
                    🦜 Gợi ý tối ưu hiệu năng
                </div>
                <div class="warning-banner-grid">
                    <div class="warning-banner-item">
                        <strong>🐢 Hệ máy CPU</strong>
                        <div class="warning-banner-content">
                            Sử dụng <b>VieNeu-TTS-0.3B-q4-gguf</b> để đạt tốc độ xử lý nhanh nhất. Nếu ưu tiên độ chính xác thì dùng <b>VieNeu-TTS-0.3B-q8-gguf</b>.
                        </div>
                    </div>
                    <div class="warning-banner-item">
                        <strong>🐆 Hệ máy GPU</strong>
                        <div class="warning-banner-content">
                            Chọn <b>VieNeu-TTS-0.3B (GPU)</b> để x2 tốc độ (độ chính xác ~95% bản gốc).<br><br>
                            ⚠️ <b>Lưu ý GPU cũ (RTX 10/20, T4):</b> Các GPU này không hỗ trợ bfloat16, nên khi dùng LMDeploy <b>bắt buộc</b> phải chọn <b>VieNeu-TTS-0.3B</b> thay vì bản VieNeu-TTS gốc.
                        </div>
                    </div>
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
                        info="Xử lý nhiều đoạn cùng lúc (chỉ áp dụng khi sử dụng GPU và đã cài đặt LMDeploy)"
                    )
                    max_batch_size_run = gr.Slider(
                        minimum=1, 
                        maximum=16, 
                        value=4, 
                        step=1, 
                        label="📊 Batch Size (Generation)",
                        info="Số lượng đoạn văn bản xử lý cùng lúc. Giá trị cao = nhanh hơn nhưng tốn VRAM hơn. Giảm xuống nếu gặp lỗi Out of Memory."
                    )
                
                with gr.Accordion("⚙️ Cài đặt nâng cao (Generation)", open=False):
                    with gr.Row():
                        temperature_slider = gr.Slider(
                            minimum=0.1, maximum=1.5, value=1.0, step=0.1,
                            label="🌡️ Temperature", 
                            info="Độ sáng tạo. Cao = đa dạng cảm xúc hơn nhưng dễ lỗi. Thấp = ổn định hơn."
                        )
                        max_chars_chunk_slider = gr.Slider(
                            minimum=128, maximum=512, value=256, step=32,
                            label="📝 Max Chars per Chunk",
                            info="Độ dài tối đa mỗi đoạn xử lý."
                        )
                
                # State to track current mode (replaces unreliable Textbox/Tabs input)
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
        
        # # --- EVENT HANDLERS ---
        # def update_info(backbone: str) -> str:
        #     return f"Streaming: {'✅' if BACKBONE_CONFIGS[backbone]['supports_streaming'] else '❌'}"
        
        # backbone_select.change(update_info, backbone_select, model_status)
        
        # Handler to show/hide Voice Cloning tab
        def on_codec_change(codec: str, current_mode: str):
            is_onnx = "onnx" in codec.lower()
            # If switching to ONNX and we are on custom mode, switch back to preset
            if is_onnx and current_mode == "custom_mode":
                return gr.update(visible=False), gr.update(selected="preset_mode"), "preset_mode"
            return gr.update(visible=not is_onnx), gr.update(), current_mode
        
        codec_select.change(
            on_codec_change, 
            inputs=[codec_select, current_mode_state], 
            outputs=[tab_custom, tabs, current_mode_state]
        )
        
        # Bind tab events to update state
        tab_preset.select(lambda: "preset_mode", outputs=current_mode_state)
        tab_custom.select(lambda: "custom_mode", outputs=current_mode_state)
        
        
        # --- Custom Model Event Handlers ---
        def on_backbone_change(choice):
            is_custom = (choice == "Custom Model")
            return gr.update(visible=is_custom)

        backbone_select.change(
            on_backbone_change,
            inputs=[backbone_select],
            outputs=[custom_model_group]
        )
        
        def on_custom_id_change(model_id):
            # Auto detect LoRA and base model
            if model_id and "lora" in model_id.lower():
                # Detect base model: if "0.3" in name -> 0.3B, else VieNeu-TTS
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
            
        custom_backbone_model_id.change(
            on_custom_id_change,
            inputs=[custom_backbone_model_id],
            outputs=[custom_backbone_base_model, custom_audio, custom_text]
        )

        btn_load.click(
            fn=load_model,
            inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb,
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
        
        # When generation starts, enable stop button
        btn_generate.click(lambda: gr.update(interactive=True), outputs=btn_stop)
        # When generation ends/stops, disable stop button
        generate_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)
        
        btn_stop.click(fn=None, cancels=[generate_event])
        btn_stop.click(lambda: (None, "⏹️ Đã dừng tạo giọng nói."), outputs=[audio_output, status_output])
        btn_stop.click(lambda: gr.update(interactive=False), outputs=btn_stop)

        # Persistence: Restore UI state on load
        demo.load(
            fn=restore_ui_state,
            outputs=[model_status, btn_generate, btn_stop]
        )

if __name__ == "__main__":
    # Cho phép override từ biến môi trường (hữu ích cho Docker)
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    # Check running in Colab
    is_on_colab = os.getenv("COLAB_RELEASE_TAG") is not None

    # Default:
    # - Colab: share=True (convenient)
    # - Docker/local: share=False (safe)
    share = env_bool("GRADIO_SHARE", default=is_on_colab)
    #
    # If server_name is "0.0.0.0" and GRADIO_SHARE is not set, disable sharing
    if server_name == "0.0.0.0" and os.getenv("GRADIO_SHARE") is None:
        share = False

    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)
