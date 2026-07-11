import sys
import io
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import gradio as gr
print("⏳ Đang khởi động VieNeu-TTS... Vui lòng chờ...")
import soundfile as sf
import tempfile
from vieneu import Vieneu
import os
import time
import numpy as np
import queue
import threading
import yaml
import uuid
from vieneu_utils.core_utils import join_audio_chunks, env_bool, get_silence_duration_v2, gaps_to_silence
from vieneu_utils.phonemize_text import phonemize_to_chunks, normalize_to_chunks, normalize_to_chunks_v3, normalize_to_chunks_v3_with_gaps
# PuncNormalizer = sea_g2p.Normalizer luôn bật punc_norm=True.
from vieneu_utils.phonemize_text import PuncNormalizer as Normalizer
import gc

# PDF text extraction
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    fitz = None

from apps.ui_utils import (
    _format_duration,
    _split_estimate_status,
    wrap_with_estimate,
    cleanup_gpu_memory,
    get_ref_text_cached,
    on_codec_change,
    validate_audio_duration,
    on_custom_id_change
)
from apps.ui_constants import (
    theme,
    css,
    head_html,
    DEFAULT_TEXT_GPU,
    DEFAULT_TEXT_TURBO,
    DEFAULT_TEXT_V3
)

# --- CONSTANTS & CONFIG ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}
except Exception as e:
    raise RuntimeError(f"Không thể đọc config.yaml: {e}")

BACKBONE_CONFIGS = _config.get("backbone_configs", {})
CODEC_CONFIGS = _config.get("codec_configs", {})

# Refilter and Simplify Configs per requirements
HAS_GPU = False
try:
    import torch
    HAS_GPU = torch.cuda.is_available() or (sys.platform == "darwin" and torch.backends.mps.is_available())
except ImportError:
    pass

filtered_backbones = {}

# VieNeu-TTS v3 Turbo (early access) — PyTorch, runs on both CPU and GPU.
# NOTE: this hardcoded `filtered_backbones` dict OVERRIDES config.yaml's
# backbone_configs (see the reassignment below), so the model list is edited here.
# int8 ĐẶT TRƯỚC → mặc định. Backbone nén int8: nhanh ~3x/frame trên CPU, nhỏ 4x,
# chất giọng đã A/B nghe đạt. fp32 giữ lại cho ai cần chất lượng tối đa (chậm hơn CPU).
filtered_backbones["VieNeu-TTS-v3-Turbo (int8)"] = {
    "repo": "pnnbao-ump/VieNeu-TTS-v3-Turbo",
    "precision": "int8",
    "supports_streaming": False,
    "description": "🆕 v3 Turbo (int8) — 48kHz, TỐI ƯU CHO CPU: nhanh nhất (backbone nén int8, ~3x/frame, nhẹ 4x). Khuyến nghị cho máy CPU. Giọng mặc định dùng speaker token; Voice Cloning clone từ audio mẫu; tag cảm xúc [cười]/[hắng giọng]/[thở dài] (thử nghiệm)."
}
filtered_backbones["VieNeu-TTS-v3-Turbo"] = {
    "repo": "pnnbao-ump/VieNeu-TTS-v3-Turbo",
    "precision": "fp32",
    "supports_streaming": False,
    "description": "v3 Turbo (fp32) — chất lượng tối đa nhưng CHẬM hơn trên CPU. Chỉ nên chọn khi cần chất lượng cao nhất; máy CPU nên dùng bản (int8)."
}

# GPU-only extras. On GPU the default is VieNeu-TTS-v2 (GPU); v3 Turbo stays the
# default (and only option) on CPU machines (the v2/v1 GGUF CPU builds were removed).
if HAS_GPU:
    filtered_backbones["VieNeu-TTS-v2 (GPU)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS-v2",
        "supports_streaming": False,
        "description": "VieNeu-TTS Version 2 - hỗ trợ song ngữ (Anh-Việt) và chế độ podcast"
    }
    filtered_backbones["VieNeu-TTS (GPU)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS",
        "supports_streaming": False,
        "description": "VieNeu-TTS Version 1 - ổn định, production-ready"
    }

BACKBONE_CONFIGS = filtered_backbones

filtered_codecs = {
    "NeuCodec (Distill)": {
        "repo": "neuphonic/distill-neucodec",
        "description": "Codec mặc định cho model GPU",
        "use_preencoded": False
    },
    "NeuCodec (ONNX)": {
        "repo": "neuphonic/neucodec-onnx-decoder-int8",
        "description": "Codec siêu nhẹ, tối ưu cho CPU (ONNX)",
        "use_preencoded": False
    },
    "VieNeu-Codec": {
        "repo": "pnnbao-ump/VieNeu-Codec",
        "description": "Codec tối ưu cho Turbo v2 (ONNX)",
        "use_preencoded": False
    }
}
CODEC_CONFIGS = filtered_codecs

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
PRESET_VOICES_CACHE = []  # List of all voices (tuples or strings)
CONV_VOICES_CACHE = []    # Filtered list for conversation (podcast=True)
MAX_SPEAKERS = 8          # Max concurrent speakers in conversation tab

# Normalizer (module-level singleton)
_text_normalizer = Normalizer()

def get_available_devices() -> list[str]:
    """Get list of available devices for current platform."""
    devices = ["Auto", "CPU"]
    
    try:
        import torch
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            devices.append("MPS")
        elif torch.cuda.is_available():
            devices.append("CUDA")
    except ImportError:
        pass

    return devices

def _supports_cloning(backbone_choice: str) -> bool:
    """Voice Cloning availability by model.

    v3+ clones directly from a sample audio; VieNeu-TTS-v2 (GPU) clones from
    audio plus a reference transcript. v1 and the CPU/Turbo builds stay
    preset-only.
    """
    c = (backbone_choice or "").lower()
    return "v3" in c or c == "vieneu-tts-v2 (gpu)"

def get_model_status_message() -> str:
    """Reconstruct status message from global state"""
    global model_loaded, tts, using_lmdeploy, current_backbone, current_codec
    if not model_loaded or tts is None:
        return "⏳ Chưa tải model."
    
    if "v2-Turbo" in (current_backbone or ""):
        backend_name = "⚡ Turbo (v2)"
    elif using_lmdeploy:
        backend_name = "🚀 LMDeploy (Optimized)"
    else:
        backend_name = "📦 Standard"
    
    # We don't track the exact device strings perfectly in global state, so we estimate
    try:
        import torch
        has_mps = torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
    except:
        has_mps = has_cuda = False

    device_info = "GPU (CUDA)" if (using_lmdeploy or "CUDA" in (current_backbone or "")) else ("MPS (Metal)" if has_mps else "Auto")
    
    if "v2-Turbo" in (current_backbone or ""):
        codec_device = "GPU/MPS" if (has_cuda or has_mps) else "CPU"
    elif "ONNX" in (current_codec or ""):
        codec_device = "CPU"
    else:
        codec_device = "GPU/MPS" if (has_cuda or has_mps) else "CPU"

    preencoded_note = ""    
    opt_info = ""
    if using_lmdeploy and hasattr(tts, 'get_optimization_stats'):
        stats = tts.get_optimization_stats()
        opt_info = (
            f"\n\n🔧 Tối ưu hóa:"
            f"\n  • Triton: {'✅' if stats['triton_enabled'] else '❌'}"
            f"\n  • Max Batch Size (Default): {stats.get('max_batch_size', 'N/A')}"
            f"\n  • Reference Cache: {stats['cached_references']} voices"
            f"\n  • Prefix Caching: ❌"
        )

    return (
        f"✅ Model đã tải thành công!\n\n"
        f"🔧 Backend: {backend_name}\n"
        f" Parrot: {current_backbone} on {device_info}\n"
        f"🎵 Codec: {current_codec} on {codec_device}{preencoded_note}{opt_info}"
    )

def restore_ui_state():
    """Update UI components based on persistence"""
    global model_loaded
    msg = get_model_status_message()
    return (
        msg, 
        gr.update(interactive=model_loaded), # btn_generate
        gr.update(interactive=model_loaded), # btn_generate_conv
        gr.update(interactive=False)         # btn_stop
    )

def should_use_lmdeploy(backbone_choice: str, device_choice: str) -> bool:
    """Determine if we should use LMDeploy backend."""
    # LMDeploy not supported on macOS
    if sys.platform == "darwin":
        return False

    # GGUF, v2-Turbo và v3 Turbo đều KHÔNG dùng LMDeploy (v3 là PyTorch, có engine riêng).
    bc = backbone_choice.lower()
    if "gguf" in bc or "v2-turbo" in bc or "v3" in bc:
        return False
    
    try:
        import torch
        if device_choice == "Auto":
            has_gpu = torch.cuda.is_available()
        elif device_choice == "CUDA":
            has_gpu = torch.cuda.is_available()
        else:
            has_gpu = False
        return has_gpu
    except ImportError:
        return False

def load_model(backbone_choice: str, codec_choice: str, device_choice: str, 
               force_lmdeploy: bool, custom_model_id: str = "", custom_base_model: str = "", 
               custom_hf_token: str = ""):
    """Load model with optimizations and max batch size control"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    lmdeploy_error_reason = None
    model_loaded = False # Ensure we don't try to use a half-loaded model
    
    # Helper for slot updates (initially no change)
    slot_no_updates = [gr.update()] * MAX_SPEAKERS

    yield (
        "⏳ Đang tải model với tối ưu hóa... Lưu ý: Quá trình này sẽ tốn thời gian. Vui lòng kiên nhẫn.",
        gr.update(interactive=False), # btn_generate
        gr.update(interactive=False), # btn_generate_conv
        gr.update(interactive=False), # btn_load
        gr.update(interactive=False), # btn_stop
        gr.update(), # voice_select
        gr.update(), gr.update(), gr.update(), gr.update(), # tab_p, tab_c, tab_sel, mode_state
        gr.update(), # conv_tab
        *slot_no_updates
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
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), # conv_tab
                    *slot_no_updates
                )
                return

            # Check if it is a LoRA to merge
            if "lora" in custom_model_id.lower():
                # Merging mode
                print(f"🔄 Detected LoRA in name. preparing merge with base: {custom_base_model}")
                if custom_base_model not in BACKBONE_CONFIGS:
                    yield (
                        f"❌ Lỗi: Base Model '{custom_base_model}' không hợp lệ.",
                        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), # conv_tab
                        *slot_no_updates
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
        use_lmdeploy = False
        
        # Override LMDeploy if custom
        if custom_loading:
             if "gguf" in backbone_config['repo'].lower() or "v2-turbo" in backbone_config['repo'].lower():
                 # GGUF must use Standard/Turbo backend
                 use_lmdeploy = False
             elif is_merged_lora:
                 # LoRA can use LMDeploy if we merge first (checked logic below) or Standard
                 use_lmdeploy = force_lmdeploy and should_use_lmdeploy(custom_base_model, device_choice)
             else:
                 # Full custom model (e.g. finetune)
                 use_lmdeploy = force_lmdeploy and should_use_lmdeploy("VieNeu-TTS (GPU)", device_choice) # Assume GPU compatible?
        # Use LMDeploy only if Force LMDeploy is set and the model is compatible
        # NOTE: For VieNeu-v2-Turbo, we handle LMDeploy inside TurboGPUVieNeuTTS class, 
        # so we set use_lmdeploy = False here to avoid generic FastVieNeuTTS loading.
        # NOTE: For custom_loading, the block above already decided use_lmdeploy correctly
        # (e.g. False for GGUF repos). Do NOT override that decision here.
        if "v2-Turbo" in backbone_choice or "v3" in backbone_choice.lower():
             # v2-Turbo có LMDeploy riêng trong class; v3 Turbo là PyTorch (engine riêng,
             # không bao giờ dùng LMDeploy) — bỏ qua FastVieNeuTTS generic.
             should_use_generic_fast = False
        elif custom_loading:
             should_use_generic_fast = False  # already handled above per repo name
        else:
             should_use_generic_fast = force_lmdeploy and should_use_lmdeploy(backbone_choice, device_choice)
             
        if should_use_generic_fast:
            use_lmdeploy = True
        
        if use_lmdeploy:
            lmdeploy_error_reason = None
            print(f"🚀 Using LMDeploy backend with optimizations")
            
            backbone_device = "cuda"
            
            if "ONNX" in codec_choice:
                codec_device = "cpu"
            else:
                try:
                    import torch
                    codec_device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    codec_device = "cpu"
            
            # Special handling for Custom LoRA + LMDeploy -> Merge & Save
            target_backbone_repo = backbone_config["repo"]
            
            if custom_loading and is_merged_lora:
                safe_name = custom_model_id.strip().replace("/", "_").replace("\\", "_").replace(":", "")
                cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "merged_models_cache", safe_name)
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
                         gr.update(interactive=False),
                         gr.update(),
                         gr.update(), gr.update(), gr.update(), gr.update(),
                         gr.update(), # conv_tab
                         *slot_no_updates
                    )
                    
                    try:
                        # Use GPU for merging if available for speed
                        # We use the Base Model specified
                        from vieneu.standard import VieNeuTTS
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
                from vieneu.fast import FastVieNeuTTS
                tts = FastVieNeuTTS(
                    backbone_repo=target_backbone_repo,
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    memory_util=0.3,
                    tp=1,
                    enable_prefix_caching=False,
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
                    gr.update(interactive=False),
                    gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), # conv_tab
                    *slot_no_updates
                )
                time.sleep(1)
                use_lmdeploy = False
                using_lmdeploy = False
        
        if not use_lmdeploy:
            print(f"📦 Using original backend")

            if device_choice == "Auto":
                repo_lower = backbone_config['repo'].lower()
                is_gguf_backbone = "gguf" in repo_lower

                if is_gguf_backbone:
                    # GGUF backbones (llama-cpp-python): Metal on Mac, CUDA on Windows/Linux
                    if sys.platform == "darwin":
                        backbone_device = "gpu"  # llama-cpp-python uses Metal via n_gpu_layers
                    else:
                        try:
                            import torch
                            backbone_device = "gpu" if torch.cuda.is_available() else "cpu"
                        except ImportError:
                            backbone_device = "cpu"
                else:
                    # PyTorch backbones (Standard, Turbo GPU): use native torch device
                    try:
                        import torch
                        if sys.platform == "darwin":
                            backbone_device = "mps" if torch.backends.mps.is_available() else "cpu"
                        else:
                            backbone_device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        backbone_device = "cpu"

                # Codec device
                if "ONNX" in codec_choice:
                    codec_device = "cpu"
                else:
                    try:
                        import torch
                        if sys.platform == "darwin":
                            codec_device = "mps" if torch.backends.mps.is_available() else "cpu"
                        else:
                            codec_device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        codec_device = "cpu"

            elif device_choice == "MPS":
                backbone_device = "mps"
                codec_device = "mps" if "ONNX" not in codec_choice else "cpu"

            else:
                backbone_device = device_choice.lower()
                codec_device = device_choice.lower()

                if "ONNX" in codec_choice:
                    codec_device = "cpu"

            if "gguf" in backbone_config['repo'].lower() and backbone_device == "cuda":
                # Only Llama-cpp (GGUF) uses the 'gpu' string for CUDA
                backbone_device = "gpu"
            
            print(f"📦 Loading model...")
            print(f"   Backbone: {backbone_config['repo']} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")
            
            if "v3-Turbo" in backbone_choice:
                # VieNeu v3 Turbo. CPU → ONNX Runtime; GPU → PyTorch. The backend is
                # auto-selected from the device inside Vieneu(mode="v3turbo"); ONNX
                # graphs are fetched from the model repo's onnx/ subfolder.
                print("   🆕 Mode: v3 Turbo (CPU=ONNX / GPU=PyTorch)")
                # Map the app's device string to what the v3 engine understands.
                v3_device = "cpu" if str(backbone_device).lower() == "cpu" else "auto"
                # precision: "int8" (mặc định, subfolder onnx_int8) | "fp32" (onnx_update).
                # Chỉ ảnh hưởng đường CPU/ONNX; trên GPU dùng PyTorch nên bỏ qua.
                v3_precision = backbone_config.get("precision", "int8")
                print(f"   🎚️  Precision: {v3_precision}")
                tts = Vieneu(
                    mode="v3turbo",
                    backbone_repo=backbone_config["repo"],
                    device=v3_device,
                    precision=v3_precision,
                    hf_token=custom_hf_token,
                )
            elif "v2-Turbo" in backbone_choice:
                # VieNeu v2 Turbo uses the dedicated backend
                print("   ⚡ Mode: Turbo")
                mode = "turbo_gpu" if "GPU" in backbone_choice else "turbo"
                tts = Vieneu(
                    mode=mode,
                    backbone_repo=backbone_config["repo"],
                    decoder_repo=codec_config["repo"],
                    device=backbone_device,
                    backend="lmdeploy" if force_lmdeploy and "GPU" in backbone_choice else "standard",
                    hf_token=custom_hf_token
                )
            else:
                from vieneu.standard import VieNeuTTS
                tts = VieNeuTTS(
                    backbone_repo=backbone_config["repo"],
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    hf_token=custom_hf_token,
                    gguf_filename=backbone_config.get("gguf_filename")
                )

            # Perform LoRA Merge if needed (ONLY for Standard Backend)
            # For LMDeploy, we handled it above by saving to disk
            if is_merged_lora and custom_loading and not using_lmdeploy:
                yield (
                    f"🔄 Đang tải và merge LoRA adapter: {custom_model_id}...",
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), # conv_tab
                    *slot_no_updates
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
                f"\n  • Prefix Caching: ❌"
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
            
            global PRESET_VOICES_CACHE, CONV_VOICES_CACHE
            PRESET_VOICES_CACHE = voices
            
            # Filter voices for conversation tab (podcast=True)
            # Handle both boolean True/False and string "True"/"False"
            def _check_podcast(v_id):
                val = tts._preset_voices.get(v_id, {}).get('podcast', True)
                if isinstance(val, str):
                    return val.strip().lower() == "true"
                return bool(val)

            CONV_VOICES_CACHE = [v for v in voices if _check_podcast(v[1])]
            
            slot_dd_update = gr.update(choices=CONV_VOICES_CACHE)
            
            # Show Standard Tabs
            tab_p = gr.update(visible=True)
            tab_c = gr.update(visible=_supports_cloning(backbone_choice))
            tab_sel = gr.update(selected="preset_mode")
            mode_state = "preset_mode"
        else:
            # Missing voices.json case
            msg = "⚠️ Không tìm thấy file voices.json. Vui lòng dùng Tab Voice Cloning."
            voice_update = gr.update(choices=[msg], value=msg, interactive=False)
            slot_dd_update = gr.update(choices=[])
            
            # Show Preset Tab (to see message) and Custom Tab
            tab_p = gr.update(visible=True)
            tab_c = gr.update(visible=_supports_cloning(backbone_choice))
            tab_sel = gr.update(selected="preset_mode")
            mode_state = "preset_mode"

        # Conversation tab: available for v2 (sequential) and v3 Turbo (batched).
        is_v2 = (backbone_choice == "VieNeu-TTS-v2 (GPU)" or backbone_choice == "VieNeu-TTS-v2 (CPU)")
        is_v3_conv = "v3" in (backbone_choice or "").lower()
        conv_tab_update = gr.update(visible=is_v2 or is_v3_conv)

        # Update all MAX_SPEAKERS slot dropdowns
        slot_updates = [slot_dd_update] * MAX_SPEAKERS

        yield (
            success_msg,
            gr.update(interactive=True), # btn_generate
            gr.update(interactive=True), # btn_generate_conv
            gr.update(interactive=True), # btn_load
            gr.update(interactive=False), # btn_stop
            voice_update,
            tab_p, tab_c, tab_sel, mode_state,
            conv_tab_update,
            *slot_updates
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
                gr.update(interactive=False), # btn_generate_conv
                gr.update(interactive=True), # btn_load
                gr.update(interactive=False), # btn_stop
                gr.update(), # voice_select
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), # conv_tab
                *slot_no_updates
            )
        else: 
            yield (
                f"❌ Lỗi khi tải model: {str(e)}",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), # conv_tab
                *slot_no_updates
            )


def resolve_voice_id(v_id: str) -> str:
    """Robustly resolve voice ID, handling both display labels and internal IDs."""
    if not v_id:
        return v_id
    
    global PRESET_VOICES_CACHE
    if not PRESET_VOICES_CACHE:
        return v_id
        
    for item in PRESET_VOICES_CACHE:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            label, value = item[0], item[1]
            if v_id == value or v_id == label:
                return value
        else:
            if v_id == item:
                return item
            
    return v_id

# --- 2. DATA & HELPERS ---

# Reading-style labels (UI) → model style keys (v3 Turbo only).
STYLE_LABEL_TO_KEY = {"Tự nhiên": "tu_nhien", "Tin tức": "tin_tuc", "Kể chuyện": "doc_truyen"}


def synthesize_speech(text: str, voice_choice: str, custom_audio, custom_text: str,
                      mode_tab: str, generation_mode: str, use_batch: bool, max_batch_size_run: int,
                      temperature: float, max_chars_chunk: int,
                      style_choice: str = "Tự nhiên", denoise_ref: bool = True, session_id: str = None):
    """Synthesis with optimization support and max batch size control"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    
    _STOP_EVENT.clear()  # Reset for new generation
    
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
    
    is_v3 = "v3" in (current_backbone or "").lower()
    style_key = STYLE_LABEL_TO_KEY.get(style_choice, "tu_nhien")
    v3_speaker_emb = None
    try:
        ref_codes = None
        ref_text_raw = ""
        v3_voice_token_id = None
        v_id = None

        if mode_tab == "preset_mode":
            if not voice_choice:
                raise ValueError("Vui lòng chọn giọng mẫu.")
            if "⚠️" in voice_choice:
                raise ValueError("Không có giọng mẫu khả dụng. Vui lòng chuyển sang Tab Voice Cloning.")

            # Use SDK method - handles caching and JSON internally
            v_id = resolve_voice_id(voice_choice)
            voice_data = tts.get_preset_voice(v_id)
            if is_v3:
                # v3 presets carry a speaker embedding + reference codes.
                v3_speaker_emb = voice_data['speaker_emb']
                ref_codes = voice_data.get('codes')
            else:
                ref_codes = voice_data['codes']
                ref_text_raw = voice_data.get('text', '')
                v3_voice_token_id = voice_data.get('reserved_id')

        elif mode_tab == "custom_mode":
            if custom_audio is None:
                raise ValueError("Vui lòng upload file Audio mẫu (Reference Audio)!")

            cb_lower = (current_backbone or "").lower()
            # Turbo v2 and v3 clone purely from audio → no reference transcript needed.
            needs_ref_text = "v2-turbo" not in cb_lower and "v3" not in cb_lower
            if needs_ref_text and (not custom_text or not custom_text.strip()):
                raise ValueError("Vui lòng nhập nội dung văn bản của Audio mẫu (Reference Text)!")

            ref_text_raw = custom_text.strip() if custom_text else ""
            if is_v3:
                # Enroll once: (optionally) denoise + trim ≤8s → speaker embedding + ref codes.
                v3_speaker_emb, ref_codes = tts.encode_reference(custom_audio, denoise=denoise_ref)
            else:
                ref_codes = tts.encode_reference(custom_audio)

        # Ensure numpy for inference
        if 'torch' in sys.modules:
            import torch
            if isinstance(ref_codes, torch.Tensor):
                ref_codes = ref_codes.cpu().numpy()

    except Exception as e:
        yield None, f"❌ Lỗi xử lý Reference Audio: {str(e)}"
        return
    
    # === STANDARD MODE ===
    if generation_mode == "Standard (Một lần)":
        # ============================ v3 TURBO BRANCH ========================
        # VieNeu-TTS v3 Turbo: split the text into chunks and run them through the
        # batched serving engine (vieneu.v3_turbo_serve) so multiple chunks share
        # each forward step (big GPU throughput win at Batch Size 32). Falls back
        # to single-utterance generation on CPU / 1 chunk / batching disabled.
        if "v3" in (current_backbone or "").lower():
            _t0 = time.time()
            yield None, "⏳ Đang tổng hợp (v3 Turbo)..."
            sr_v3 = getattr(tts, "sample_rate", 48000)
            try:
                from vieneu_utils.phonemize_text import phonemize_text_with_emotions

                # Chia chunk theo TEXT đã normalize (giống v2-gpu, không vụn), giữ
                # inline cues; phonemize TỪNG chunk khi dựng request.
                v3_chunks, v3_gaps = normalize_to_chunks_v3_with_gaps(raw_text, max_chars=max_chars_chunk)
                v3_bs = max(1, int(max_batch_size_run)) if use_batch else 1
                v3_engine_dev = getattr(getattr(tts, "engine", None), "device", None)
                v3_can_batch = (
                    v3_bs > 1 and len(v3_chunks) > 1
                    and v3_engine_dev is not None and v3_engine_dev.type == "cuda"
                )

                if v3_can_batch:
                    from vieneu.v3_turbo_serve import V3TurboBatchEngine
                    if getattr(tts, "_v3_batch_engine", None) is None:
                        tts._v3_batch_engine = V3TurboBatchEngine(tts.engine)
                    v3_wavs = []
                    for i in range(0, len(v3_chunks), v3_bs):
                        if _STOP_EVENT.is_set():
                            yield None, "⏹️ Đã dừng tạo giọng nói."
                            return
                        group = v3_chunks[i:i + v3_bs]
                        yield None, f"⚡ v3 Turbo: lô {i // v3_bs + 1} ({len(group)} đoạn, batch size {v3_bs})..."
                        # group là các TEXT chunk -> phonemize từng cái (giữ inline cues).
                        reqs = [{"phonemes": phonemize_text_with_emotions(c), "speaker_emb": v3_speaker_emb,
                                 "ref_codes": ref_codes, "style": style_key, "use_ref_codes": True} for c in group]
                        v3_wavs.extend(tts._v3_batch_engine.generate_batch(
                            reqs, temperature=temperature, max_new_frames=300))
                    wav = join_audio_chunks(v3_wavs, sr=sr_v3, silence_ps=gaps_to_silence(v3_gaps))
                else:
                    # CPU (ONNX) hoặc GPU khi tắt batch: xử lý TUẦN TỰ từng đoạn.
                    # Gọi engine 1 lần / đoạn (giống nhánh batch nhưng tuần tự) để có
                    # ĐÚNG 1 wav / đoạn -> khớp với v3_gaps khi join, đồng thời báo cho
                    # người dùng đang xử lý đến đoạn nào + ước tính thời gian còn lại —
                    # quan trọng trên CPU vì mỗi đoạn có thể mất nhiều giây.
                    # (KHÔNG enumerate tts.infer(): infer() trả về 1 mảng đã ghép sẵn,
                    #  lặp qua nó sẽ ra từng sample numpy.float32 -> len() ném lỗi.)
                    total_v3 = len(v3_chunks)
                    v3_wavs = []
                    chunk_durations = []
                    last_t = time.time()
                    for i, chunk in enumerate(v3_chunks):
                        if _STOP_EVENT.is_set():
                            yield None, "⏹️ Đã dừng tạo giọng nói."
                            return
                        yield None, f"⏳ v3 Turbo: Đang xử lý đoạn {i + 1}/{total_v3}..."
                        ph = phonemize_text_with_emotions(chunk)
                        chunk_wav = tts.engine.infer(
                            phonemes=ph, speaker_emb=v3_speaker_emb, ref_codes=ref_codes,
                            style=style_key, use_ref_codes=True,
                            temperature=temperature, max_new_frames=300)
                        now = time.time()
                        chunk_durations.append(now - last_t)
                        last_t = now
                        if chunk_wav is not None and len(chunk_wav) > 0:
                            v3_wavs.append(chunk_wav)
                        done = i + 1
                        if done < total_v3:
                            avg = sum(chunk_durations) / len(chunk_durations)
                            eta = avg * (total_v3 - done)
                            yield None, (
                                f"⏳ v3 Turbo: Đã xong {done}/{total_v3} đoạn "
                                f"(ước tính còn lại: {_format_duration(eta)})... "
                                f"đang xử lý đoạn {done + 1}/{total_v3}"
                            )
                    wav = join_audio_chunks(v3_wavs, sr=sr_v3, silence_ps=gaps_to_silence(v3_gaps))
            except Exception as e:
                yield None, f"❌ Lỗi tổng hợp (v3 Turbo): {str(e)}"
                return
            if wav is None or len(wav) == 0:
                yield None, "❌ Không sinh được audio nào."
                return
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, wav, sr_v3)
                out_path_v3 = tmp.name
            _dt = time.time() - _t0
            _spd = f", Tốc độ: {len(wav)/sr_v3/_dt:.2f}x realtime" if _dt > 0 else ""
            yield out_path_v3, f"✅ Hoàn tất! (v3 Turbo, Thời gian: {_dt:.2f}s{_spd})"
            cleanup_gpu_memory()
            return
        # ========================== end v3 TURBO BRANCH ======================

        backend_name = "LMDeploy" if using_lmdeploy else "Standard"

        is_v2_turbo = "v2-Turbo" in (current_backbone or "")
        
        if is_v2_turbo:
            text_chunks = phonemize_to_chunks(raw_text, max_chars=max_chars_chunk)
        else:
            # Chia chunk SAU normalize: normalize cả văn bản trước rồi mới cắt theo
            # độ dài ĐÃ chuẩn hóa (tránh chunk phình quá max_chars khi norm mở rộng).
            text_chunks = normalize_to_chunks(raw_text, max_chars=max_chars_chunk)
            
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
            if is_v2_turbo:
                # Sequential processing with progress updates
                total_chunks = len(text_chunks)
                for i, chunk in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return
                    yield None, f"⚡ Turbo v2: Đang xử lý đoạn {i+1}/{total_chunks}..."
                    
                    chunk_wav = tts.infer(
                        chunk.text, 
                        ref_codes=ref_codes, 
                        temperature=temperature,
                        max_chars=max_chars_chunk,
                        skip_normalize=True,
                        skip_phonemize=True
                    )
                    
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_wavs.append(chunk_wav)
                        # Add silence between Gradio-level chunks for Turbo
                        if i < total_chunks - 1:
                            sil_dur = get_silence_duration_v2(chunk)
                            sil_wav = np.zeros(int(sr * sil_dur), dtype=np.float32)
                            all_wavs.append(sil_wav)
            
            # Use batch processing if enabled and using LMDeploy (for v1)
            elif use_batch and using_lmdeploy and hasattr(tts, 'infer_batch') and total_chunks > 1:
                # Process in mini-batches to allow cancellation between batches
                num_batches = (total_chunks + max_batch_size_run - 1) // max_batch_size_run
                total_batch_duration = 0.0
                completed_batches = 0
                
                for i in range(0, total_chunks, max_batch_size_run):
                    if _STOP_EVENT.is_set():
                        print("🛑 Synthesis stopped during batch processing.")
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return
                    
                    batch_idx = i // max_batch_size_run
                    estimate_info = ""
                    if completed_batches > 0:
                        average_batch_duration = total_batch_duration / completed_batches
                        estimated_total = average_batch_duration * num_batches
                        estimated_remaining = average_batch_duration * max(0, num_batches - batch_idx)
                        estimate_info = (
                            f" | Ước tính còn lại: {_format_duration(estimated_remaining)}"
                            f" / tổng: {_format_duration(estimated_total)}"
                        )
                    yield None, f"⚡ Đang xử lý batch {batch_idx+1}/{num_batches} (đoạn {i+1}-{min(i+max_batch_size_run, total_chunks)}){estimate_info}..."
                    
                    current_batch = text_chunks[i : i + max_batch_size_run]
                    batch_start_time = time.time()
                    batch_wavs = tts.infer_batch(
                        current_batch, 
                        ref_codes=ref_codes, 
                        ref_text=ref_text_raw,
                        max_batch_size=max_batch_size_run,
                        temperature=temperature,
                        skip_normalize=True
                    )
                    batch_duration = time.time() - batch_start_time
                    total_batch_duration += batch_duration
                    completed_batches += 1
                    average_batch_duration = total_batch_duration / completed_batches
                    estimated_total = average_batch_duration * num_batches
                    estimated_remaining = average_batch_duration * max(0, num_batches - completed_batches)
                    for chunk_wav in batch_wavs:
                        if chunk_wav is not None and len(chunk_wav) > 0:
                            all_wavs.append(chunk_wav)
                    yield None, (
                        f"✅ Xong batch {batch_idx+1}/{num_batches} "
                        f"(trung bình batch: {_format_duration(average_batch_duration)}, "
                        f"ước tính còn lại: {_format_duration(estimated_remaining)}, "
                        f"tổng: {_format_duration(estimated_total)})"
                    )

            else:
                # Sequential processing (PyTorch or GGUF v1)
                for i, chunk in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return
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
            silence_p = 0.15 if not is_v2_turbo else 0.0 # Turbo adds silence internally
            final_wav = join_audio_chunks(all_wavs, sr=sr, silence_p=silence_p)
            
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
            
        except Exception as e:
            # Check for CUDA OOM specifically if torch is loaded
            if 'torch' in sys.modules:
                import torch
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    cleanup_gpu_memory()
                    yield None, (
                        f"❌ GPU hết VRAM! Hãy thử:\n"
                        f"• Giảm Max Batch Size (hiện tại: {tts.max_batch_size if hasattr(tts, 'max_batch_size') else 'N/A'})\n"
                        f"• Giảm độ dài văn bản\n\n"
                        f"Chi tiết: {str(e)}"
                    )
                    return
            
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
        
        is_v2_turbo = "v2-Turbo" in (current_backbone or "")
        if is_v2_turbo:
            text_chunks = phonemize_to_chunks(raw_text, max_chars=max_chars_chunk)
        else:
            # Chia chunk SAU normalize: normalize cả văn bản trước rồi mới cắt theo
            # độ dài ĐÃ chuẩn hóa (tránh chunk phình quá max_chars khi norm mở rộng).
            text_chunks = normalize_to_chunks(raw_text, max_chars=max_chars_chunk)
        
        def producer_thread():
            nonlocal error_msg
            try:
                previous_tail = None
                
                for i, chunk_text in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        break
                    
                    if is_v2_turbo:
                        stream_gen = tts.infer_stream(
                            chunk_text.text,
                            ref_codes=ref_codes, 
                            temperature=temperature,
                            max_chars=max_chars_chunk,
                            skip_normalize=True,
                            skip_phonemize=True,
                            emotion_tag=""
                        )
                    else:
                        stream_gen = tts.infer_stream(
                            chunk_text, 
                            ref_codes=ref_codes, 
                            ref_text=ref_text_raw,
                            temperature=temperature,
                            max_chars=max_chars_chunk,
                            skip_normalize=True,
                            emotion_tag=""
                        )
                    
                    for part_idx, audio_part in enumerate(stream_gen):
                        if _STOP_EVENT.is_set():
                            break
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
                            
                    # Add silence between chunks for Turbo v2
                    if is_v2_turbo and i < len(text_chunks) - 1:
                        sil_dur = get_silence_duration_v2(chunk_text)
                        sil_wav = np.zeros(int(sr * sil_dur), dtype=np.float32)
                        audio_queue.put((sr, sil_wav))
                
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

synthesize_speech_with_estimate = wrap_with_estimate(synthesize_speech)

def synthesize_conversation_with_empty_estimate(*args):
    for audio_path, status in synthesize_conversation(*args):
        yield audio_path, status, ""

# --- CANCELLATION ---
# threading.Event is a mutable object: never reassigned, always the same reference.
# All threads share the exact same object — no scoping/serialization issues.
_STOP_EVENT = threading.Event()

# --- 3. CONVERSATION LOGIC ---

def _synthesize_conversation_v3(lines, mapping, temperature, max_chars_chunk, silence_duration):
    """v3 Turbo conversation: batch the WHOLE conversation at batch size 32.

    Unlike the v2 path (sequential, one turn at a time), every turn is split into
    chunks and ALL chunks across ALL turns are flattened into one request list, then
    fed to the V3 batched engine in groups of 32 — regardless of which speaker each
    chunk belongs to (each request carries its own speaker ref_codes + reserved
    token). Wavs are reassembled per turn (in order) and joined with inter-turn
    silence. Yields ``(audio_path, status)`` like the v2 path. 48 kHz.
    """
    global tts
    from collections import defaultdict
    from vieneu_utils.core_utils import join_audio_chunks, gaps_to_silence
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions, normalize_to_chunks_v3_with_gaps
    # NOTE: KHÔNG import vieneu.v3_turbo_serve ở đây — module đó import torch ở cấp
    # module, nên trên bản cài CPU/macOS không-torch (ONNX) sẽ lỗi "No module named
    # 'torch'". Chỉ import bên trong nhánh CUDA bên dưới (nơi thực sự cần batch engine).

    sr = getattr(tts, "sample_rate", 48000)
    t0 = time.time()

    # Resolve each speaker → (speaker_emb, ref_codes), cached per speaker.
    def _voice_for(spk_name):
        cfg = mapping.get(spk_name.lower())
        v_id = (cfg or {}).get('voice') or tts._default_voice
        try:
            vd = tts.get_preset_voice(v_id)
        except Exception:
            vd = tts.get_preset_voice(tts._default_voice)
        emb = vd.get('speaker_emb')
        rc = vd.get('codes')
        if 'torch' in sys.modules:
            import torch
            if isinstance(rc, torch.Tensor):
                rc = rc.cpu().numpy()
        return (np.asarray(emb, dtype=np.float32) if emb is not None else None,
                np.asarray(rc) if rc is not None else None)

    # CPU (ONNX) has no batched engine → run sequentially, one turn at a time.
    dev = getattr(getattr(tts, "engine", None), "device", None)
    is_cuda = dev is not None and getattr(dev, "type", None) == "cuda"
    if not is_cuda:
        all_wavs = []
        for li, line in enumerate(lines):
            if _STOP_EVENT.is_set():
                yield None, "⏹️ Đã dừng hội thoại."
                return
            cfg = mapping.get(line['speaker'].lower())
            v_id = (cfg or {}).get('voice') or tts._default_voice
            yield None, f"⏳ [{li+1}/{len(lines)}] {line['speaker']}: {line['text'][:30]}..."
            try:
                wav = tts.infer(line['text'], voice=v_id, style="tu_nhien",
                                temperature=temperature, max_chars=max_chars_chunk)
            except Exception as e:
                print(f"❌ Lỗi câu {li+1}: {e}")
                continue
            if wav is not None and len(wav):
                all_wavs.append(wav)
                if li < len(lines) - 1 and silence_duration > 0:
                    all_wavs.append(np.zeros(int(sr * silence_duration), dtype=np.float32))
        if not all_wavs:
            yield None, "❌ Không thể tạo được âm thanh nào!"
            return
        yield None, "🪄 Đang ghép nối âm thanh..."
        final_wav = np.concatenate(all_wavs)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, final_wav, sr)
            yield tmp.name, f"✅ Hoàn tất hội thoại! ({len(lines)} câu, {time.time()-t0:.1f}s, CPU tuần tự)"
        cleanup_gpu_memory()
        return

    voice_cache = {}
    reqs, req_line = [], []
    line_gaps = {}
    for li, line in enumerate(lines):
        key = line['speaker'].lower()
        if key not in voice_cache:
            voice_cache[key] = _voice_for(line['speaker'])
        spk_emb, ref_codes = voice_cache[key]
        # Chia chunk theo TEXT đã normalize (giống v2-gpu), giữ inline cues, rồi
        # phonemize từng chunk. Hội thoại luôn dùng phong cách Tự nhiên.
        line_chunks, line_gaps[li] = normalize_to_chunks_v3_with_gaps(line['text'], max_chars=max_chars_chunk)
        for chunk in line_chunks:
            reqs.append({"phonemes": phonemize_text_with_emotions(chunk),
                         "speaker_emb": spk_emb, "ref_codes": ref_codes,
                         "style": "tu_nhien", "use_ref_codes": True})
            req_line.append(li)

    if not reqs:
        yield None, "❌ Không có lời thoại để tổng hợp."
        return

    if getattr(tts, "_v3_batch_engine", None) is None:
        # Chỉ tới đây khi chạy trên CUDA → torch chắc chắn có sẵn.
        from vieneu.v3_turbo_serve import V3TurboBatchEngine
        tts._v3_batch_engine = V3TurboBatchEngine(tts.engine)

    BS = 32
    total_batches = (len(reqs) + BS - 1) // BS
    wavs_flat = []
    for bi, i in enumerate(range(0, len(reqs), BS)):
        if _STOP_EVENT.is_set():
            yield None, "⏹️ Đã dừng hội thoại."
            return
        group = reqs[i:i + BS]
        yield None, f"⚡ v3 Turbo hội thoại: lô {bi + 1}/{total_batches} ({len(group)} đoạn, batch 32)..."
        wavs_flat.extend(tts._v3_batch_engine.generate_batch(
            group, temperature=temperature, max_new_frames=300))

    # Reassemble per turn (in order), then join turns with inter-turn silence.
    by_line = defaultdict(list)
    for w, li in zip(wavs_flat, req_line):
        by_line[li].append(w)

    all_wavs = []
    for li in range(len(lines)):
        lw = join_audio_chunks(by_line[li], sr=sr, silence_ps=gaps_to_silence(line_gaps.get(li, []))) if by_line[li] else None
        if lw is None or len(lw) == 0:
            continue
        all_wavs.append(lw)
        if li < len(lines) - 1 and silence_duration > 0:
            all_wavs.append(np.zeros(int(sr * silence_duration), dtype=np.float32))

    if not all_wavs:
        yield None, "❌ Không thể tạo được âm thanh nào!"
        return

    yield None, "🪄 Đang ghép nối âm thanh..."
    final_wav = np.concatenate(all_wavs)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, final_wav, sr)
        elapsed = time.time() - t0
        yield tmp.name, f"✅ Hoàn tất hội thoại! ({len(lines)} câu, {len(reqs)} đoạn, {elapsed:.1f}s, batch 32)"
    cleanup_gpu_memory()


def synthesize_conversation(
    script_text: str,
    *args
):
    """
    Synthesizes multi-speaker conversation from a script.

    Gradio passes speaker name boxes and voice dropdowns as individual positional args.
    Layout: args[0..MAX_SPEAKERS-1] = speaker names, args[MAX_SPEAKERS..2*MAX_SPEAKERS-1] = voice IDs,
    args[2*MAX_SPEAKERS] = silence_duration, args[2*MAX_SPEAKERS+1] = temperature,
    args[2*MAX_SPEAKERS+2] = max_chars_chunk, args[2*MAX_SPEAKERS+3] = session_id
    """
    speaker_names     = list(args[:MAX_SPEAKERS])
    speaker_voices    = list(args[MAX_SPEAKERS:MAX_SPEAKERS*2])
    silence_duration  = args[MAX_SPEAKERS * 2]
    temperature       = args[MAX_SPEAKERS * 2 + 1]
    max_chars_chunk   = args[MAX_SPEAKERS * 2 + 2]
    session_id        = args[MAX_SPEAKERS * 2 + 3] if len(args) > MAX_SPEAKERS * 2 + 3 else None

    global tts, model_loaded, using_lmdeploy
    
    _STOP_EVENT.clear()
    
    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return
        
    if not script_text or script_text.strip() == "":
        yield None, "⚠️ Vui lòng nhập kịch bản hội thoại!"
        return

    # 1. Parse Script
    lines = []
    for line in script_text.strip().split('\n'):
        if not line.strip(): continue
        if ':' in line:
            parts = line.split(':', 1)
            lines.append({'speaker': parts[0].strip(), 'text': parts[1].strip()})
        else:
            if lines:
                lines[-1]['text'] += " " + line.strip()
            else:
                lines.append({'speaker': 'Narrator', 'text': line.strip()})

    if not lines:
        yield None, "⚠️ Không tìm thấy lời thoại hợp lệ (định dạng Nhân vật: Lời thoại)!"
        return

    # 2. Build Speaker Mapping from individual slot components
    mapping = {}
    for name, voice in zip(speaker_names, speaker_voices):
        name = str(name).strip() if name else ""
        if not name: continue
        # Use lowercase key for robust matching
        v_id = resolve_voice_id(str(voice)) if voice else ""
        mapping[name.lower()] = {
            'type': 'Preset',
            'voice': v_id,
            'ref_text': ''
        }

    # 2b. v3 Turbo: batch toàn bộ hội thoại (bs=32), bất kể speaker thay đổi.
    if "v3" in (current_backbone or "").lower():
        yield from _synthesize_conversation_v3(
            lines, mapping, temperature, max_chars_chunk, silence_duration
        )
        return

    # 3. Process Each Line
    all_wavs = []
    sr = 24000
    total_lines = len(lines)
    
    yield None, f"🎭 Đang khởi tạo hội thoại ({total_lines} câu)..."
    
    start_time = time.time()
    
    try:
        for i, line in enumerate(lines):
            if _STOP_EVENT.is_set():
                yield None, "⏹️ Đã dừng hội thoại."
                return
            spk_name = line['speaker']
            text = line['text']
            
            yield None, f"⏳ [{i+1}/{total_lines}] {spk_name}: {text[:30]}..."
            
            # Determine voice
            ref_codes = None
            ref_text_val = None
            current_voice_obj = None
            
            # Case-insensitive lookup
            config = mapping.get(spk_name.lower())
            
            if not config:
                print(f"  ⚠️ Character '{spk_name}' not found in mapping. Fallback to default.")
                # Fallback to default if speaker not mapped
                try:
                    # Get default voice data
                    default_v_id = tts._default_voice
                    if not default_v_id:
                        dv_list = tts.list_preset_voices()
                        if dv_list:
                            first = dv_list[0]
                            default_v_id = first[1] if isinstance(first, tuple) else first
                    
                    if default_v_id:
                        current_voice_obj = tts.get_preset_voice(default_v_id)
                        ref_codes = current_voice_obj['codes']
                        ref_text_val = current_voice_obj['text']
                except Exception as e:
                    print(f"  ❌ Fallback failed: {e}")
            else:
                try:
                    v_id = config['voice']
                    if config['type'] == "Preset":
                        current_voice_obj = tts.get_preset_voice(v_id)
                        if current_voice_obj and 'codes' in current_voice_obj:
                            ref_codes = current_voice_obj['codes']
                            ref_text_val = current_voice_obj['text']
                        else:
                            print(f"  ❌ Could not find codes for voice '{v_id}'")
                    else: # Custom
                        if v_id and os.path.exists(v_id):
                            ref_codes = tts.encode_reference(v_id)
                            ref_text_val = config.get('ref_text', '')
                            current_voice_obj = {'codes': ref_codes, 'text': ref_text_val}
                            print(f"  🦜 Using custom voice for '{spk_name}'")
                except Exception as e:
                    print(f"  ❌ Lỗi nạp giọng cho {spk_name} (ID: {config.get('voice')}): {e}")
            
            # Ensure numpy for inference
            if 'torch' in sys.modules:
                import torch
                if isinstance(ref_codes, torch.Tensor):
                    ref_codes = ref_codes.cpu().numpy()

            # Infer audio
            try:
                wav = tts.infer(
                    text,
                    voice=current_voice_obj, # Use full voice object
                    ref_codes=ref_codes,     # Fallback if object not supported
                    ref_text=ref_text_val,
                    temperature=temperature,
                    max_chars=max_chars_chunk,
                    emotion_tag="<|emotion_0|>" # Emotion tag for conversation
                )
                
                all_wavs.append(wav)
                
                # Add silence between turns
                if i < total_lines - 1 and silence_duration > 0:
                    silence_len = int(sr * silence_duration)
                    silence = np.zeros(silence_len)
                    all_wavs.append(silence)
                    
            except Exception as e:
                print(f"❌ Lỗi tổng hợp câu {i+1}: {e}")
                continue

        if not all_wavs:
            yield None, "❌ Không thể tạo được âm thanh nào!"
            return

        # 4. Merge and Output
        yield None, "🪄 Đang ghép nối âm thanh..."
        final_wav = np.concatenate(all_wavs)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, final_wav, sr)
            elapsed = time.time() - start_time
            yield tmp.name, f"✅ Hoàn tất hội thoại! ({total_lines} câu, xử lý trong {elapsed:.1f}s)"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"❌ Lỗi hệ thống: {str(e)}"

def extract_speakers_from_script(script):
    """Find unique speakers and return gr.update() lists for the 8 slot components."""
    global CONV_VOICES_CACHE
    if not script:
        # Hide all slots
        name_updates = [gr.update(value="", visible=False)] * MAX_SPEAKERS
        dd_updates   = [gr.update(value=None, visible=False)] * MAX_SPEAKERS
        row_updates  = [gr.update(visible=False)] * MAX_SPEAKERS
        return name_updates + dd_updates + row_updates

    speakers = []
    seen = set()
    for line in script.strip().split('\n'):
        if ':' in line:
            s = line.split(':', 1)[0].strip()
            if s and s not in seen:
                seen.add(s)
                speakers.append(s)

    # Auto-match each speaker name to a preset voice
    def _best_match(name):
        if not CONV_VOICES_CACHE:
            return None
        
        name_l = name.lower()
        
        # 0. Manual overrides for specific common names
        overrides = {
            "phương": "Trúc Ly",
            "dũng": "Thanh Bình",
            "hùng": "Thái Sơn"
        }
        if name_l in overrides:
            target = overrides[name_l].lower()
            for v in CONV_VOICES_CACHE:
                label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
                if target in label.lower() or target in value.lower():
                    return value

        # 1. Try to find name in labels or values
        for v in CONV_VOICES_CACHE:
            label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
            if name_l == label.lower() or name_l == value.lower():
                return value
        
        # 2. Fuzzy match (contains)
        for v in CONV_VOICES_CACHE:
            label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
            if name_l in label.lower() or name_l in value.lower() or label.lower() in name_l or value.lower() in name_l:
                return value
        
        # 3. Default to first voice if no match
        first_voice = CONV_VOICES_CACHE[0]
        return first_voice[1] if isinstance(first_voice, tuple) else first_voice

    name_updates, dd_updates, row_updates = [], [], []
    for i in range(MAX_SPEAKERS):
        if i < len(speakers):
            name_updates.append(gr.update(value=speakers[i], visible=True))
            dd_updates.append(gr.update(value=_best_match(speakers[i]), choices=CONV_VOICES_CACHE, visible=True))
            row_updates.append(gr.update(visible=True))
        else:
            name_updates.append(gr.update(value="", visible=False))
            dd_updates.append(gr.update(value=None, choices=CONV_VOICES_CACHE, visible=False))
            row_updates.append(gr.update(visible=False))

    return name_updates + dd_updates + row_updates

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using PyMuPDF (fitz)."""
    global HAS_FITZ, fitz
    if not HAS_FITZ:
        return "⚠️ PyMuPDF chưa được cài đặt. Vui lòng chạy: pip install PyMuPDF"
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text and text.strip():
                pages_text.append(text.strip())
        doc.close()
        if not pages_text:
            return "⚠️ Không tìm thấy văn bản trong file PDF."
        full_text = "\n\n".join(pages_text)
        return full_text
    except Exception as e:
        return f"⚠️ Lỗi khi đọc PDF: {str(e)}"

EXAMPLES_LIST = [
    ["Về miền Tây không chỉ để ngắm nhìn sông nước hữu tình, mà còn để cảm nhận tấm chân tình của người dân nơi đây.", "Vĩnh (nam miền Nam)"],
    ["Hà Nội những ngày vào thu mang một vẻ đẹp trầm mặc và cổ kính đến lạ thường.", "Bình (nam miền Bắc)"],
]

with gr.Blocks(theme=theme, css=css, title="VieNeu-TTS", head=head_html) as demo:
    # Session ID for cancellation tracking
    session_id_state = gr.State("")

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
            <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2" target="_blank" class="model-card-link">VieNeu-TTS-v2</a>
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
                # --- BACKBONE & CODEC DEFAULT LOGIC ---
                # GPU users default to VieNeu-TTS-v3-Turbo (GPU); CPU-only users get v3 Turbo
                # (the only CPU backbone). v3 (GPU) is registered solely when HAS_GPU.
                # int8 là entry đầu tiên → mặc định trên cả CPU lẫn GPU (trên GPU dùng
                # PyTorch nên int8/fp32 như nhau; trên CPU int8 nhanh nhất).
                default_backbone = list(BACKBONE_CONFIGS.keys())[0]
                
                # Default parameters based on backbone
                if "v3" in default_backbone.lower():
                    default_codec = "VieNeu-Codec"
                    default_temp = 0.8
                    default_text = DEFAULT_TEXT_V3
                elif "Turbo" in default_backbone:
                    default_codec = "VieNeu-Codec"
                    default_temp = 0.4
                    default_text = DEFAULT_TEXT_TURBO
                elif "(CPU)" in default_backbone:
                    default_codec = "NeuCodec (ONNX)"
                    default_temp = 0.7
                    default_text = DEFAULT_TEXT_GPU
                else:
                    default_codec = "NeuCodec (Distill)" if "NeuCodec (Distill)" in CODEC_CONFIGS else list(CODEC_CONFIGS.keys())[0]
                    default_temp = 0.7
                    default_text = DEFAULT_TEXT_GPU

                # v3 Turbo batches chunks through the serving engine → default 32.
                # Must be set at creation: v3 is the default backbone, so the
                # on_backbone_change handler (which also sets 32) never fires on load.
                default_batch_size = 32 if "v3" in default_backbone.lower() else 4

                backbone_select = gr.Dropdown(
                    list(BACKBONE_CONFIGS.keys()) + ["Custom Model"],
                    value=default_backbone,
                    label="🦜 Backbone"
                )
                codec_select = gr.Dropdown(
                    list(CODEC_CONFIGS.keys()), 
                    value=default_codec, 
                    label="🎵 Codec",
                    interactive=False
                )
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
                base_model_choices = [k for k in BACKBONE_CONFIGS.keys() if "turbo" not in k.lower() and k != "Custom Model"]
                custom_backbone_base_model = gr.Dropdown(
                    base_model_choices,
                    label="🔗 Base Model (cho LoRA)",
                    value=base_model_choices[0] if base_model_choices else None,
                    visible=False,
                    info="Model gốc để merge với LoRA (GPU Only)",
                    scale=1
                )
            
            with gr.Row():
                use_lmdeploy_cb = gr.Checkbox(
                    value=True,
                    label="🚀 Optimize with LMDeploy (Khuyên dùng cho NVIDIA GPU)",
                    info="Tick nếu bạn dùng GPU để tăng tốc độ tổng hợp đáng kể.",
                    visible="v3" not in default_backbone.lower(),  # v3 Turbo (PyTorch) không dùng LMDeploy
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
                        <strong>🐆 Hệ máy GPU</strong>
                        <div class="warning-banner-content">
                            <b>VieNeu-TTS-v3-Turbo (early access)</b> đã được phát hành để dùng thử trước, đã hỗ trợ các tag cảm xúc `[cười]` `[hắng giọng]` `[thở dài]`, tuy nhiên những tính năng này vẫn đang được thử nghiệm và chưa thực sự ổn định, có thể sẽ xảy ra lỗi không mong muốn, nếu có lỗi các bạn hãy thông báo với chúng tôi tại: https://discord.com/invite/yJt8kzjzWZ. Trong trường hợp bạn cần sự ổn định hãy sử dụng <b>VieNeu-TTS-v2 (GPU)</b>. 
                        </div>
                    </div>
                    <div class="warning-banner-item" style="background: #dcfce7; border-color: #86efac;">
                        <strong style="color: #15803d;">🐢 Hệ máy CPU</strong>
                        <div class="warning-banner-content" style="color: #166534;">
                            Máy <b>CPU</b> nên dùng bản <b>VieNeu-TTS-v3-Turbo (int8)</b> để tốc độ tối đa. Chuyển sang <b>VieNeu-TTS-v3-Turbo</b> nếu cần chất lượng cao hơn (nhưng chậm hơn trên CPU).
                        </div>
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.85rem; color: #92400e; border-top: 1px dashed #fcd34d; padding-top: 8px;">
                    💡 <b>Mẹo:</b> Nếu máy bạn có GPU mà không thấy các phiên bản GPU hãy xem lại cách cài đặt uv sync --group gpu
                </div>
            </div>
            """)

            gr.Markdown(
                "🆕 **VieNeu-TTS-v3-Turbo (early access)** đã được phát hành để **dùng thử trước** — "
                "48kHz, **hỗ trợ Voice Cloning** (tính năng clone chỉ có từ **v3** trở lên; v1/v2 không hỗ trợ). "
                "Bản **v3 đầy đủ** sẽ ra mắt trong **vài tuần tới**.\n\n"
                "🎭 v3 Turbo đã **hỗ trợ các tag cảm xúc** `[cười]` `[hắng giọng]` `[thở dài]` "
                "(chèn trực tiếp vào văn bản) — nhưng tính năng này **đang thử nghiệm và chưa thực sự ổn định**."
            )

            btn_load = gr.Button("🔄 Tải Model", variant="primary")
            model_status = gr.Markdown("⏳ Chưa tải model.")
        
        with gr.Row(elem_classes="container"):
            # --- INPUT ---
            with gr.Column(scale=3):
                with gr.Tabs() as main_input_tabs:
                    # --- TAB 1: SINGLE SPEAKER ---
                    with gr.Tab("🦜 Đọc truyện", id="single_tab") as single_tab:
                        with gr.Accordion("📄 Tải lên PDF để trích xuất văn bản", open=False):
                            gr.Markdown(
                                "Tải lên file PDF, văn bản sẽ được tự động trích xuất và điền vào ô bên dưới. "
                                "Bạn có thể chỉnh sửa lại trước khi tạo audio."
                            )
                            with gr.Row():
                                pdf_upload = gr.File(
                                    label="📄 Chọn file PDF",
                                    file_types=[".pdf"],
                                    file_count="single",
                                    type="filepath",
                                    scale=3,
                                )
                                btn_extract_pdf = gr.Button("📄 Trích xuất văn bản", variant="secondary", scale=1, min_width=150)
                            pdf_status = gr.Markdown(visible=False)
                        text_input = gr.Textbox(
                            label=f"Văn bản",
                            lines=8,
                            value=default_text,
                        )
                        
                        with gr.Tabs() as tabs:
                            with gr.TabItem("👤 Preset", id="preset_mode") as tab_preset:
                                voice_select = gr.Dropdown(choices=[], value=None, label="Giọng mẫu", allow_custom_value=True)
                            
                            # Voice cloning is only available on v3+ models. Hidden by
                            # default and toggled on by on_backbone_change when a v3
                            # model is selected.
                            with gr.TabItem("🦜 Voice Cloning", id="custom_mode", visible=False) as tab_custom:
                                # Initial clone-tab state must match the DEFAULT backbone:
                                # on_backbone_change only fires when the dropdown changes, so a
                                # v2-GPU default would otherwise keep v3's "no transcript" copy
                                # and a hidden reference-text box (-> false "missing ref text").
                                _default_is_v2_gpu = (default_backbone == "VieNeu-TTS-v2 (GPU)")
                                clone_info_md = gr.Markdown(
                                    "ℹ️ **Voice Cloning (VieNeu-TTS v2).** Tải lên audio mẫu 3–5 giây "
                                    "và **nhập đúng nội dung** của audio đó (kể cả dấu câu) — v2 cần "
                                    "reference transcript để clone giọng."
                                    if _default_is_v2_gpu else
                                    "ℹ️ **Voice Cloning (VieNeu-TTS v3).** Chỉ cần tải lên audio mẫu "
                                    "3–5 giây; v3 clone trực tiếp từ audio, không cần nhập nội dung."
                                )
                                with gr.Group(visible=True) as cloning_elements_group:
                                    custom_audio = gr.Audio(label="Audio giọng mẫu (3-5 giây) (.wav)", type="filepath")
                                    cloning_warning_msg = gr.Markdown(visible=False, elem_id="cloning-warning")
                                    denoise_checkbox = gr.Checkbox(
                                        value=True, label="🔇 Denoise audio mẫu",
                                        info="Khử nhiễu nền + chuẩn hoá audio mẫu trước khi clone (khuyến nghị). Audio dài hơn 8 giây sẽ được cắt ngắn.",
                                    )
                                    # v3 clones from audio only — the reference transcript box
                                    # is hidden for v3 (toggled by on_backbone_change).
                                    custom_text = gr.Textbox(label="Nội dung audio mẫu - vui lòng gõ đúng nội dung của audio mẫu - kể cả dấu câu vì model rất nhạy cảm với dấu câu (.,?!)", visible=_default_is_v2_gpu)
                                    gr.Examples(
                                        examples=[
                                            [os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "audio_ref", "example.wav"), "Ví dụ 2. Tính trung bình của dãy số."],
                                            [os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "audio_ref", "example_2.wav"), "Trên thực tế, các nghi ngờ đã bắt đầu xuất hiện."],
                                            [os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "audio_ref", "example_3.wav"), "Cậu có nhìn thấy không?"],
                                            [os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "audio_ref", "example_4.wav"), "Tết là dịp mọi người háo hức đón chào một năm mới với nhiều hy vọng và mong ước."]
                                        ],
                                        inputs=[custom_audio, custom_text],
                                        label="Ví dụ mẫu để thử nghiệm clone giọng"
                                    )
                                    
                                    gr.Markdown("""
                                    **💡 Mẹo nhỏ:** Nếu kết quả Zero-shot Voice Cloning chưa như ý, bạn hãy cân nhắc **Finetune (LoRA)** để đạt chất lượng tốt nhất. 
                                    Hướng dẫn chi tiết có tại file: `finetune/README.md` hoặc xem trên [GitHub](https://github.com/pnnbao97/VieNeu-TTS/tree/main/finetune).
                                    """)
                        
                        style_dropdown = gr.Dropdown(
                            ["Tự nhiên", "Tin tức", "Kể chuyện"],
                            value="Tự nhiên",
                            label="🎭 Phong cách đọc",
                            info="Phong cách giọng đọc (áp dụng cho VieNeu-TTS v3).",
                        )

                        generation_mode = gr.Radio(
                            ["Standard (Một lần)"],
                            value="Standard (Một lần)",
                            label="Chế độ sinh"
                        )
                        btn_generate = gr.Button("🎵 Bắt đầu", variant="primary", scale=2, interactive=False)

                    # --- TAB 2: MULTI-SPEAKER CONVERSATION ---
                    with gr.Tab("🎭 Hội thoại", id="conv_tab", visible=False) as conv_tab:
                        conv_script_input = gr.Textbox(
                            label="Kịch bản hội thoại",
                            placeholder="Phương: Chào mọi người, mình là Phương...",
                            lines=10,
                            elem_classes="script-box",
                            value='Phương: Chào mọi người, mình là Phương. Hôm nay team có một announcement cực lớn về VieNeu-TTS Version 2. Đồng hành cùng mình là anh Dũng và Hùng. Hi guys!\n\nDũng: Yo, chào cả nhà. Mình sẽ đi thẳng vào technical side của bản nâng cấp này để mọi người có cái nhìn deep hơn nhé.\n\nHùng: Chào mọi người. Thật sự V2 là một huge milestone. Nó phá vỡ rào cản của những công cụ đọc văn bản khô khan, hướng tới một sự natural communication đúng nghĩa.\n\nPhương: Correct! Và bất ngờ nhất là: nãy giờ mọi người đang nghe bản demo được tạo ra 100% bằng VieNeu-TTS V2 đấy. Tụi mình đều là sản phẩm của AI hết. Amazing, right?\n\nDũng: Đỉnh thật sự! Tiện đây Hùng share thêm về cái nội công bên trong của model này đi.\n\nHùng: Chắc chắn rồi. Model được train trên 10000 hours audio chất lượng cao, nên nó hỗ trợ code-switching Anh Việt cực mượt, tự nhiên như podcast. Đặc biệt, dự án này hoàn toàn open-source để cộng đồng cùng phát triển.\n\nDũng: Về hiệu năng thì khỏi bàn. Khi test trên GPU quốc dân RTX 3060, tốc độ sinh audio nhanh gấp 10 lần realtime. Và đừng lo, nếu bạn không có card đồ hỏa xịn, tụi mình có sẵn bản CPU version để ai cũng có thể tiếp cận được.\n\nPhương: Tốc độ cực nhanh, hỗ trợ đa nền tảng và hoàn toàn miễn phí. Mọi người hãy cùng trải nghiệm nhé!'
                        )
                        
                        with gr.Row():
                            btn_detect_speakers = gr.Button("🔍 Quét nhân vật", size="sm", variant="secondary")
                            silence_slider = gr.Slider(minimum=0, maximum=3, value=0.1, step=0.1, label="⏱️ Khoảng lặng (giây)")

                        gr.Markdown("### 🎭 Cấu hình giọng đọc")
                        gr.Markdown("*Nhấn **Quét nhân vật** để tự động phát hiện và ánh xạ giọng đọc. Tải model trước để có danh sách giọng.*")

                        # Pre-build MAX_SPEAKERS speaker slot rows
                        speaker_name_boxes = []
                        speaker_voice_dds  = []
                        speaker_slot_rows  = []

                        for _i in range(MAX_SPEAKERS):
                            # Mặc định cho 3 nhân vật đầu tiên theo yêu cầu
                            _default_name = ""
                            _default_voice = None
                            _row_visible = False
                            
                            if _i == 0:
                                _default_name = "Phương"
                                _default_voice = "Ly"
                                _row_visible = True
                            elif _i == 1:
                                _default_name = "Dũng"
                                _default_voice = "Binh"
                                _row_visible = True
                            elif _i == 2:
                                _default_name = "Hùng"
                                _default_voice = "Sơn"
                                _row_visible = True
                            elif _i < 2:
                                _default_name = f"Nhân vật {_i+1}"
                                _row_visible = True

                            with gr.Row(visible=_row_visible) as _row:
                                _name = gr.Textbox(
                                    value=_default_name,
                                    label="👤 Nhân vật",
                                    interactive=False,
                                    scale=1,
                                    min_width=120
                                )
                                _dd = gr.Dropdown(
                                    choices=PRESET_VOICES_CACHE,
                                    value=_default_voice,
                                    label="🎤 Giọng đọc",
                                    interactive=True,
                                    scale=3,
                                    allow_custom_value=True
                                )
                            speaker_slot_rows.append(_row)
                            speaker_name_boxes.append(_name)
                            speaker_voice_dds.append(_dd)
                        
                        btn_generate_conv = gr.Button("🎭 Bắt đầu hội thoại", variant="primary", interactive=False)

                # Global Generation Settings
                with gr.Row():
                    use_batch = gr.Checkbox(
                        value=True, 
                        label="⚡ Batch Processing",
                        info="Xử lý nhiều đoạn cùng lúc (chỉ áp dụng khi sử dụng GPU và đã cài đặt LMDeploy)"
                    )
                    max_batch_size_run = gr.Slider(
                        minimum=1,
                        maximum=32,
                        value=default_batch_size,
                        step=1,
                        label="📊 Batch Size (Generation)",
                        info="Số lượng đoạn văn bản xử lý cùng lúc. Giá trị cao = nhanh hơn nhưng tốn VRAM hơn. Giảm xuống nếu gặp lỗi Out of Memory."
                    )
                
                with gr.Accordion("⚙️ Cài đặt nâng cao (Generation)", open=False):
                    with gr.Row():
                        temperature_slider = gr.Slider(
                            minimum=0.1, maximum=1.5, value=default_temp, step=0.1,
                            label="🌡️ Temperature", 
                            info="Độ sáng tạo. Cao = đa dạng cảm xúc hơn nhưng dễ lỗi. Thấp = ổn định hơn."
                        )
                        max_chars_chunk_slider = gr.Slider(
                            minimum=128, maximum=512,
                            value=256, step=32,
                            label="📝 Max Chars per Chunk",
                            info="Độ dài tối đa mỗi đoạn xử lý (mặc định: 256)."
                        )
                
                # State to track current mode
                current_mode_state = gr.State("preset_mode")
                
                with gr.Row():
                    btn_stop = gr.Button("⏹️ Dừng", variant="stop", scale=1, interactive=False)
            
            # --- OUTPUT ---
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Kết quả",
                    type="filepath",
                    autoplay=True
                )
                with gr.Group():
                    status_output = gr.Textbox(
                        label="Trạng thái", 
                        elem_classes="status-box",
                        lines=2,
                        max_lines=10,
                        show_copy_button=True
                    )
                with gr.Group():
                    estimate_output = gr.Textbox(
                        label="Ước tính thời gian",
                        elem_classes="estimate-box",
                        lines=2,
                        max_lines=4,
                        show_copy_button=True
                    )
                download_btn = gr.DownloadButton(
                    "📥 Tải xuống file Audio",
                    variant="primary",
                    visible=False,
                    elem_classes="download-btn"
                )
                gr.Markdown("<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>🔒 Audio được đóng dấu bản quyền ẩn (Watermarker) để bảo mật và định danh AI.</div>")
        
        codec_select.change(
            on_codec_change, 
            inputs=[codec_select, current_mode_state], 
            outputs=[tab_custom, tabs, current_mode_state]
        )
        
        # Bind tab events to update state
        tab_preset.select(lambda: "preset_mode", outputs=current_mode_state)
        tab_custom.select(lambda: "custom_mode", outputs=current_mode_state)
        
        custom_audio.change(validate_audio_duration, inputs=[custom_audio], outputs=[cloning_warning_msg])
        
        # --- Custom Model Event Handlers ---

        def on_backbone_change(choice):
            is_custom = (choice == "Custom Model")
            is_v3 = "v3" in (choice or "").lower()
            is_v2_gpu = (choice == "VieNeu-TTS-v2 (GPU)")
            # Voice Cloning: v3 clones from audio only; v2 (GPU) clones from
            # audio + a reference transcript. Both expose the cloning tab.
            clone_ok = is_v3 or is_v2_gpu
            print(f"   🔄 Backbone changed to: {choice}")
            
            # 1. Device logic
            # Allow hardware acceleration (MPS/CUDA/Auto) for all GPU models AND Turbo (GGUF) models
            is_hw_accel_supported = "(GPU)" in choice or "v2-Turbo" in choice or "v3" in choice.lower() or is_custom
            
            if is_hw_accel_supported:
                dev_choices = get_available_devices()
                initial_dev = "Auto"
            else:
                dev_choices = ["CPU"]
                initial_dev = "CPU"
            
            # 2. Parameter logic
            if is_v3:
                # v3 Turbo uses its own MOSS codec (PyTorch); 0.8 khớp bản tham chiếu.
                codec_update = gr.update(value="VieNeu-Codec", interactive=False)
                text_update = gr.update(value=DEFAULT_TEXT_V3)
                temp_update = gr.update(value=0.8)
            elif "Turbo" in choice:
                codec_update = gr.update(value="VieNeu-Codec", interactive=False)
                text_update = gr.update(value=DEFAULT_TEXT_TURBO)
                temp_update = gr.update(value=0.4)
            elif "(CPU)" in choice:
                codec_update = gr.update(value="NeuCodec (ONNX)", interactive=False)
                text_update = gr.update(value=DEFAULT_TEXT_GPU)
                temp_update = gr.update(value=0.7)
            else:
                codec_update = gr.update(value="NeuCodec (Distill)", interactive=False)
                text_update = gr.update(value=DEFAULT_TEXT_GPU)
                temp_update = gr.update(value=0.7)

            # Reference-transcript box + info text differ between v2 and v3 clone.
            if is_v2_gpu:
                clone_info_update = gr.update(value=(
                    "ℹ️ **Voice Cloning (VieNeu-TTS v2).** Tải lên audio mẫu 3–5 giây "
                    "và **nhập đúng nội dung** của audio đó (kể cả dấu câu) — v2 cần "
                    "reference transcript để clone giọng."
                ))
            else:
                clone_info_update = gr.update(value=(
                    "ℹ️ **Voice Cloning (VieNeu-TTS v3).** Chỉ cần tải lên audio mẫu "
                    "3–5 giây; v3 clone trực tiếp từ audio, không cần nhập nội dung."
                ))

            return (
                gr.update(visible=is_custom),
                codec_update,
                text_update,
                temp_update,
                gr.update(choices=dev_choices, value=initial_dev),
                gr.update(visible=clone_ok),   # cloning_elements_group
                gr.update(visible=clone_ok),   # tab_custom — clone tab (v3 + v2 GPU)
                gr.update(value=32 if is_v3 else 4),  # max_batch_size_run — v3 batches chunks
                gr.update(visible=not is_v3),  # use_lmdeploy_cb — irrelevant for v3 (PyTorch, no LMDeploy)
                gr.update(visible=is_v2_gpu),  # custom_text — only v2 needs a reference transcript
                clone_info_update,             # clone_info_md
                gr.update(value=256),  # max_chars_chunk_slider
            )

        backbone_select.change(
            on_backbone_change,
            inputs=[backbone_select],
            outputs=[
                custom_model_group,
                codec_select,
                text_input,
                temperature_slider,
                device_choice,
                cloning_elements_group,
                tab_custom,
                max_batch_size_run,
                use_lmdeploy_cb,
                custom_text,
                clone_info_md,
                max_chars_chunk_slider,
            ]
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
            outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                     tab_preset, tab_custom, tabs, current_mode_state,
                     conv_tab,
                     *speaker_voice_dds]
        )
        
        # --- PDF Upload Event Handlers ---
        def on_pdf_upload(pdf_file):
            """Extract text from uploaded PDF and populate text input."""
            if not pdf_file:
                return gr.update(), gr.update(visible=False)
            extracted = extract_text_from_pdf(pdf_file)
            if extracted.startswith("⚠️"):
                return gr.update(), gr.update(value=extracted, visible=True)
            char_count = len(extracted)
            return (
                gr.update(value=extracted),
                gr.update(
                    value=f"✅ Đã trích xuất **{char_count:,}** ký tự từ file PDF. Bạn có thể chỉnh sửa văn bản bên dưới.",
                    visible=True
                )
            )

        pdf_upload.change(
            fn=on_pdf_upload,
            inputs=[pdf_upload],
            outputs=[text_input, pdf_status]
        )
        btn_extract_pdf.click(
            fn=on_pdf_upload,
            inputs=[pdf_upload],
            outputs=[text_input, pdf_status]
        )

        # --- Conversation Event Handlers ---
        # Scan speakers → update all 8 slot rows/names/dropdowns
        btn_detect_speakers.click(
            fn=extract_speakers_from_script,
            inputs=[conv_script_input],
            outputs=speaker_name_boxes + speaker_voice_dds + speaker_slot_rows
        )
        
        conv_gen_event = btn_generate_conv.click(
            fn=synthesize_conversation_with_empty_estimate,
            inputs=[conv_script_input,
                    *speaker_name_boxes,
                    *speaker_voice_dds,
                    silence_slider, temperature_slider, max_chars_chunk_slider,
                    session_id_state],
            outputs=[audio_output, status_output, estimate_output]
        )
        btn_generate_conv.click(lambda: gr.update(visible=False), outputs=[download_btn])
        btn_generate_conv.click(lambda: gr.update(interactive=True), outputs=btn_stop)
        conv_gen_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)

        # --- Auto-adjust Temperature on Tab Switch ---
        conv_tab.select(
            fn=lambda bb: gr.update(value=0.8 if "v3" in (bb or "").lower() else 1.0),
            inputs=backbone_select,
            outputs=temperature_slider
        )
        single_tab.select(
            fn=lambda bb: gr.update(value=0.8 if "v3" in (bb or "").lower() else default_temp),
            inputs=backbone_select,
            outputs=temperature_slider
        )
        
        # --- Standard Generation Handlers ---
        gen_event = btn_generate.click(
            fn=synthesize_speech_with_estimate,
            inputs=[text_input, voice_select, custom_audio, custom_text, current_mode_state,
                    generation_mode, use_batch, max_batch_size_run,
                    temperature_slider, max_chars_chunk_slider,
                    style_dropdown, denoise_checkbox, session_id_state],
            outputs=[audio_output, status_output, estimate_output]
        )
        btn_generate.click(lambda: gr.update(visible=False), outputs=[download_btn])
        btn_generate.click(lambda: gr.update(interactive=True), outputs=btn_stop)
        gen_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)

        # --- Stop Button ---
        def request_stop():
            print("🛑 STOP REQUESTED via button click.")
            _STOP_EVENT.set()
            return None, "⏹️ Đã dừng tạo giọng nói.", "", gr.update(interactive=False)

        # Handler: set stop event + update UI
        # Note: We avoid cancels= here to prevent internal Gradio KeyError crashes,
        # relying instead on the frequent _STOP_EVENT.is_set() checks in the code.
        btn_stop.click(fn=request_stop, outputs=[audio_output, status_output, estimate_output, btn_stop])

        # --- Download Button Event Handlers ---
        def on_audio_generated(audio_path):
            """Show download button when audio is generated."""
            if audio_path and os.path.exists(audio_path):
                return gr.update(value=audio_path, visible=True)
            return gr.update(visible=False)

        # Connect to generation events (must be after gen_event/conv_gen_event are defined)
        gen_event.then(
            fn=on_audio_generated,
            inputs=[audio_output],
            outputs=[download_btn]
        )
        conv_gen_event.then(
            fn=on_audio_generated,
            inputs=[audio_output],
            outputs=[download_btn]
        )
        # Also connect the stop button to hide download
        btn_stop.click(
            fn=lambda: gr.update(visible=False),
            outputs=[download_btn]
        )

        # Persistence: Restore UI state on load
        demo.load(
            fn=restore_ui_state,
            outputs=[model_status, btn_generate, btn_generate_conv, btn_stop]
        )

def main():
    # Cho phép override từ biến môi trường (hữu ích cho Docker)
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    # Check running in Colab
    is_on_colab = os.getenv("COLAB_RELEASE_TAG") is not None

    # Default:
    # - Colab: share=True (convenient)
    # - Docker/local: share=False (safe)
    share = env_bool("GRADIO_SHARE", default=is_on_colab)
    
    # If server_name is "0.0.0.0" and GRADIO_SHARE is not set, disable sharing
    if server_name == "0.0.0.0" and os.getenv("GRADIO_SHARE") is None:
        share = False

    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)

if __name__ == "__main__":
    main()
