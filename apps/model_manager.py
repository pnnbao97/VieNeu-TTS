import os
import sys
import gc
import time
import json
import logging
import yaml
import torch
import gradio as gr
from vieneu import Vieneu

logger = logging.getLogger("Vieneu.ModelManager")

# --- CONSTANTS & CONFIG ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}
except Exception as e:
    raise RuntimeError(f"Không thể đọc config.yaml: {e}")

# Refilter and Simplify Configs per requirements
HAS_GPU = False
try:
    import torch
    HAS_GPU = torch.cuda.is_available() or (sys.platform == "darwin" and torch.backends.mps.is_available())
except ImportError:
    pass

filtered_backbones = {}
if HAS_GPU:
    filtered_backbones["VieNeu-TTS (GPU)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS",
        "supports_streaming": False,
        "description": "⭐ Chất lượng cao nhất, yêu cầu GPU"
    }
    filtered_backbones["VieNeu-TTS-0.3B (GPU)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS-0.3B",
        "supports_streaming": False,
        "description": "⚡ Bản 0.3B tối ưu cho GPU, rất nhanh"
    }
    filtered_backbones["VieNeu-TTS-v2-Turbo (GPU)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS-v2-Turbo",
        "supports_streaming": False,
        "description": "🚀 Turbo v2 (GPU): Hỗ trợ bilingual (Anh-Việt), tối ưu cho GPU"
    }

filtered_backbones["VieNeu-TTS-v2-Turbo (CPU)"] = {
    "repo": "pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF",
    "supports_streaming": True,
    "description": "🚀 Turbo v2: Siêu nhanh, tối ưu tuyệt đối cho CPU & Thiết bị yếu"
}

BACKBONE_CONFIGS = filtered_backbones

filtered_codecs = {
    "NeuCodec (Distill)": {
        "repo": "neuphonic/distill-neucodec",
        "description": "Codec mặc định cho model GPU",
        "use_preencoded": False
    },
    "VieNeu-Codec": {
        "repo": "pnnbao-ump/VieNeu-Codec",
        "description": "Codec tối ưu cho Turbo v2 (ONNX)",
        "use_preencoded": False
    }
}
CODEC_CONFIGS = filtered_codecs

# Global model instance
tts = None
current_backbone = None
current_codec = None
model_loaded = False
using_lmdeploy = False

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

def get_model_status_message(is_xpu: bool = False) -> str:
    """Reconstruct status message from global state"""
    global model_loaded, tts, using_lmdeploy, current_backbone, current_codec
    if not model_loaded or tts is None:
        return "⏳ Chưa tải model."

    if is_xpu:
        backend_name = "🚀 Intel XPU (BFloat16 native)"
        device_info = "XPU"
        codec_device = "CPU" if "ONNX" in (current_codec or "") else "XPU"
    else:
        if "v2-Turbo" in (current_backbone or ""):
            backend_name = "⚡ Turbo (v2)"
        elif using_lmdeploy:
            backend_name = "🚀 LMDeploy (Optimized)"
        else:
            backend_name = "📦 Standard"

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
        f"🎵 Codec: {current_codec} on {codec_device}{opt_info}"
    )

def restore_ui_state(is_xpu: bool = False):
    """Update UI components based on persistence"""
    global model_loaded
    msg = get_model_status_message(is_xpu=is_xpu)
    return (
        msg,
        gr.update(interactive=model_loaded), # btn_generate
        gr.update(interactive=False)         # btn_stop
    )

def should_use_lmdeploy(backbone_choice: str, device_choice: str) -> bool:
    """Determine if we should use LMDeploy backend."""
    if sys.platform == "darwin":
        return False
    if "gguf" in backbone_choice.lower() or "v2-turbo" in backbone_choice.lower():
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

def cleanup_gpu_memory():
    """Aggressively cleanup GPU memory"""
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    gc.collect()

def load_model(backbone_choice: str, codec_choice: str, device_choice: str,
               force_lmdeploy: bool = False, custom_model_id: str = "", custom_base_model: str = "",
               custom_hf_token: str = "", is_xpu: bool = False):
    """Load model with optimizations and max batch size control"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    lmdeploy_error_reason = None
    model_loaded = False

    loading_msg = "⏳ Đang tải model lên Intel XPU... Vui lòng chờ trong giây lát..." if is_xpu else \
                  "⏳ Đang tải model với tối ưu hóa... Lưu ý: Quá trình này sẽ tốn thời gian. Vui lòng kiên nhẫn."

    yield (
        loading_msg,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update()
    )

    try:
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
                    gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update()
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
        use_lmdeploy = False

        if custom_loading:
             if "gguf" in backbone_config['repo'].lower() or "v2-turbo" in backbone_config['repo'].lower():
                 use_lmdeploy = False
             elif is_merged_lora:
                 use_lmdeploy = force_lmdeploy and should_use_lmdeploy(custom_base_model, device_choice)
             else:
                 use_lmdeploy = force_lmdeploy and should_use_lmdeploy("VieNeu-TTS (GPU)", device_choice)

        if is_xpu:
            should_use_generic_fast = False
        elif "v2-Turbo" in backbone_choice:
             should_use_generic_fast = False
        elif custom_loading:
             should_use_generic_fast = False
        else:
             should_use_generic_fast = force_lmdeploy and should_use_lmdeploy(backbone_choice, device_choice)

        if not is_xpu and should_use_generic_fast:
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

            target_backbone_repo = backbone_config["repo"]

            if custom_loading and is_merged_lora:
                safe_name = custom_model_id.strip().replace("/", "_").replace("\\", "_").replace(":", "")
                cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "merged_models_cache", safe_name)
                target_backbone_repo = os.path.abspath(cache_dir)

                if not os.path.exists(cache_dir) or not os.path.exists(os.path.join(cache_dir, "vocab.json")):
                    print(f"🔄 Merging LoRA for LMDeploy optimization: {cache_dir}")
                    yield (
                         f"⏳ Đang merge và lưu model LoRA để tối ưu cho LMDeploy (thao tác này chỉ chạy một lần)...",
                         gr.update(interactive=False),
                         gr.update(interactive=False),
                         gr.update(interactive=False),
                         gr.update(),
                         gr.update(), gr.update(), gr.update(), gr.update()
                    )

                    try:
                        from vieneu.standard import VieNeuTTS
                        base_repo = BACKBONE_CONFIGS[custom_base_model]["repo"]
                        merge_device = "cuda" if torch.cuda.is_available() else "cpu"

                        temp_tts = VieNeuTTS(
                            backbone_repo=base_repo,
                            backbone_device=merge_device,
                            codec_repo=codec_config["repo"],
                            codec_device="cpu",
                            hf_token=custom_hf_token
                        )
                        temp_tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)

                        if hasattr(temp_tts.backbone, "merge_and_unload"):
                            temp_tts.backbone = temp_tts.backbone.merge_and_unload()

                        temp_tts.backbone.save_pretrained(cache_dir)
                        temp_tts.tokenizer.save_pretrained(cache_dir)

                        try:
                            from transformers import AutoTokenizer
                            slow_tokenizer = AutoTokenizer.from_pretrained(base_repo, use_fast=False)
                            slow_tokenizer.save_pretrained(cache_dir)
                        except Exception:
                            pass

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
                    except Exception as e:
                        raise RuntimeError(f"Failed to merge & save LoRA for LMDeploy: {e}")

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
            except Exception as e:
                error_str = str(e)
                if "$env:CUDA_PATH" in error_str:
                    lmdeploy_error_reason = "Không tìm thấy biến môi trường CUDA_PATH. Vui lòng cài đặt NVIDIA GPU Computing Toolkit."
                else:
                    lmdeploy_error_reason = f"{error_str}"

                yield (
                    f"⚠️ LMDeploy Init Error: {lmdeploy_error_reason}. Đang loading model với backend mặc định - tốc độ chậm hơn so với lmdeploy...",
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                )
                time.sleep(1)
                use_lmdeploy = False
                using_lmdeploy = False

        if not use_lmdeploy:
            print(f"📦 Using original backend")
            if is_xpu:
                backbone_device = "xpu"
                codec_device = "cpu" if "ONNX" in codec_choice else "xpu"
            elif device_choice == "Auto":
                repo_lower = backbone_config['repo'].lower()
                is_gguf_backbone = "gguf" in repo_lower
                if is_gguf_backbone:
                    if sys.platform == "darwin":
                        backbone_device = "gpu"
                    else:
                        try:
                            import torch
                            backbone_device = "gpu" if torch.cuda.is_available() else "cpu"
                        except ImportError:
                            backbone_device = "cpu"
                else:
                    try:
                        import torch
                        if sys.platform == "darwin":
                            backbone_device = "mps" if torch.backends.mps.is_available() else "cpu"
                        else:
                            backbone_device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        backbone_device = "cpu"

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
                backbone_device = "gpu"

            if is_xpu:
                from vieneu.core_xpu import XPUVieNeuTTS
                tts = XPUVieNeuTTS(
                    backbone_repo=backbone_config["repo"],
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    hf_token=custom_hf_token
                )
            elif "v2-Turbo" in backbone_choice:
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
                    hf_token=custom_hf_token
                )

            if is_merged_lora and custom_loading and not using_lmdeploy:
                merge_msg = f"🔄 Đang tải và merge LoRA adapter: {custom_model_id} trên XPU..." if is_xpu else \
                            f"🔄 Đang tải và merge LoRA adapter: {custom_model_id}..."
                yield (
                    merge_msg,
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update()
                )
                try:
                    tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                    if hasattr(tts, 'backbone') and hasattr(tts.backbone, 'merge_and_unload'):
                        tts.backbone = tts.backbone.merge_and_unload()
                        tts._lora_loaded = False
                        tts._current_lora_repo = None
                except Exception as e:
                     raise RuntimeError(f"Failed to merge LoRA: {e}")

            using_lmdeploy = False

        current_backbone = backbone_choice
        current_codec = codec_choice
        model_loaded = True

        success_msg = get_model_status_message(is_xpu=is_xpu)
        if lmdeploy_error_reason:
            success_msg += f"\n\n⚠️ **Cảnh báo:** Không thể kích hoạt LMDeploy do lỗi: {lmdeploy_error_reason}"

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
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False),
            voice_update, tab_p, tab_c, tab_sel, mode_state
        )

    except Exception as e:
        model_loaded = False
        using_lmdeploy = False
        error_msg = str(e)
        if "$env:CUDA_PATH" in error_msg:
            error_msg = "Không tìm thấy CUDA_PATH. Vui lòng cài đặt NVIDIA GPU Toolkit."
        yield (
            f"❌ Lỗi: {error_msg}",
            gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        )
