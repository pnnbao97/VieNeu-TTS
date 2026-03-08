import gradio as gr
import soundfile as sf
import torch
import os
import sys
import time
import numpy as np
import queue
import threading
import gc
from vieneu import VieNeuTTS, FastVieNeuTTS
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks, env_bool
from vieneu_utils.normalize_text import VietnameseTTSNormalizer

from .ui_shared import (
    BACKBONE_CONFIGS,
    CODEC_CONFIGS,
    cleanup_gpu_memory,
    on_codec_change,
    validate_audio_duration,
    on_backbone_change,
    on_custom_id_change,
    resolve_reference_voice,
    get_backbone_config,
    wrap_tempfile_sf_write,
    UI_THEME,
    COMMON_CSS,
    HEADER_HTML,
    FAVICON_HEAD,
)

print("⏳ Đang khởi động VieNeu-TTS... Vui lòng chờ...")

# --- 1. MODEL CONFIGURATION ---
# Global model instance
tts = None
current_backbone = None
current_codec = None
model_loaded = False
using_lmdeploy = False

# Normalizer (module-level singleton)
_text_normalizer = VietnameseTTSNormalizer()


def get_available_devices() -> list[str]:
    """Get list of available devices for current platform."""
    devices = ["Auto", "CPU"]
    if sys.platform == "darwin":
        if torch.backends.mps.is_available():
            devices.append("MPS")
    else:
        if torch.cuda.is_available():
            devices.append("CUDA")
    return devices


def get_model_status_message() -> str:
    global model_loaded, tts, using_lmdeploy, current_backbone, current_codec
    if not model_loaded or tts is None:
        return "⏳ Chưa tải model."

    codec_config = CODEC_CONFIGS.get(current_codec, {})
    backend_name = "🚀 LMDeploy (Optimized)" if using_lmdeploy else "📦 Standard"
    preencoded_note = (
        "\n⚠️ Codec ONNX không hỗ trợ chức năng clone giọng nói." if codec_config.get("use_preencoded") else ""
    )

    opt_info = ""
    if using_lmdeploy and hasattr(tts, "get_optimization_stats"):
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
    global model_loaded
    msg = get_model_status_message()
    return (
        msg,
        gr.update(interactive=model_loaded),  # btn_generate
        gr.update(interactive=False),  # btn_stop
    )


def should_use_lmdeploy(backbone_choice: str, device_choice: str) -> bool:
    if sys.platform == "darwin":
        return False
    if "gguf" in backbone_choice.lower():
        return False
    if device_choice == "Auto" or device_choice == "CUDA":
        return torch.cuda.is_available()
    return False


def load_model(
    backbone_choice: str,
    codec_choice: str,
    device_choice: str,
    force_lmdeploy: bool,
    custom_model_id: str = "",
    custom_base_model: str = "",
    custom_hf_token: str = "",
):
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    lmdeploy_error_reason = None
    model_loaded = False

    yield (
        "⏳ Đang tải model với tối ưu hóa... Lưu ý: Quá trình này sẽ tốn thời gian. Vui lòng kiên nhẫn.",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )

    try:
        if tts is not None:
            tts = None
            cleanup_gpu_memory()

        try:
            backbone_config, custom_loading, is_merged_lora = get_backbone_config(
                backbone_choice, custom_model_id, custom_base_model
            )
        except ValueError as ve:
            yield (
                f"❌ Lỗi: {ve}",
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )
            return

        codec_config = CODEC_CONFIGS[codec_choice]

        if custom_loading:
            if "gguf" in backbone_config["repo"].lower():
                use_lmdeploy = False
            elif is_merged_lora:
                use_lmdeploy = force_lmdeploy and should_use_lmdeploy(custom_base_model, device_choice)
            else:
                use_lmdeploy = force_lmdeploy and should_use_lmdeploy("VieNeu-TTS (GPU)", device_choice)
        else:
            use_lmdeploy = force_lmdeploy and should_use_lmdeploy(backbone_choice, device_choice)

        if use_lmdeploy:
            print("🚀 Using LMDeploy backend with optimizations")
            backbone_device = "cuda"
            codec_device = "cpu" if "ONNX" in codec_choice else ("cuda" if torch.cuda.is_available() else "cpu")

            target_backbone_repo = backbone_config["repo"]
            if custom_loading and is_merged_lora:
                safe_name = custom_model_id.strip().replace("/", "_").replace("\\", "_").replace(":", "")
                cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "merged_models_cache", safe_name)
                target_backbone_repo = os.path.abspath(cache_dir)

                if not os.path.exists(cache_dir) or not os.path.exists(os.path.join(cache_dir, "vocab.json")):
                    yield (
                        "⏳ Đang merge và lưu model LoRA để tối ưu cho LMDeploy (thao tác này chỉ chạy một lần)...",
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                    try:
                        base_repo = BACKBONE_CONFIGS[custom_base_model]["repo"]
                        merge_device = "cuda" if torch.cuda.is_available() else "cpu"
                        temp_tts = VieNeuTTS(
                            base_repo,
                            backbone_device=merge_device,
                            codec_repo=codec_config["repo"],
                            codec_device="cpu",
                            hf_token=custom_hf_token,
                        )
                        temp_tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                        if hasattr(temp_tts.backbone, "merge_and_unload"):
                            temp_tts.backbone = temp_tts.backbone.merge_and_unload()
                        temp_tts.backbone.save_pretrained(cache_dir)
                        temp_tts.tokenizer.save_pretrained(cache_dir)
                        try:
                            from transformers import AutoTokenizer
                            AutoTokenizer.from_pretrained(base_repo, use_fast=False).save_pretrained(cache_dir)
                        except Exception: pass
                        import json
                        with open(os.path.join(cache_dir, "voices.json"), "w", encoding="utf-8") as f:
                            json.dump({
                                "meta": {"note": "Auto-generated during LoRA merge"},
                                "default_voice": temp_tts._default_voice,
                                "presets": temp_tts._preset_voices,
                            }, f, ensure_ascii=False, indent=2)
                        del temp_tts
                        cleanup_gpu_memory()
                    except Exception as e:
                        raise RuntimeError(f"Failed to merge & save LoRA for LMDeploy: {e}")

            try:
                tts = FastVieNeuTTS(
                    backbone_repo=target_backbone_repo,
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    memory_util=0.3, tp=1, enable_prefix_caching=True, enable_triton=True, hf_token=custom_hf_token,
                )
                using_lmdeploy = True
            except Exception as e:
                error_str = str(e)
                lmdeploy_error_reason = "CUDA_PATH error" if "$env:CUDA_PATH" in error_str else error_str
                yield (
                    f"⚠️ LMDeploy Init Error: {lmdeploy_error_reason}. Chuyển sang Standard backend...",
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                )
                time.sleep(1)
                use_lmdeploy = False
                using_lmdeploy = False

        if not use_lmdeploy:
            if device_choice == "Auto":
                if "gguf" in backbone_config["repo"].lower():
                    backbone_device = "gpu"
                else:
                    backbone_device = "mps" if sys.platform == "darwin" and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
                codec_device = "cpu" if "ONNX" in codec_choice else backbone_device
            elif device_choice == "MPS":
                backbone_device = codec_device = "mps"
                if "ONNX" in codec_choice: codec_device = "cpu"
            else:
                backbone_device = codec_device = device_choice.lower()
                if "ONNX" in codec_choice: codec_device = "cpu"

            if "gguf" in backbone_config["repo"].lower() and backbone_device == "cuda":
                backbone_device = "gpu"

            tts = VieNeuTTS(
                backbone_repo=backbone_config["repo"],
                backbone_device=backbone_device,
                codec_repo=codec_config["repo"],
                codec_device=codec_device,
                hf_token=custom_hf_token,
            )
            if is_merged_lora and custom_loading:
                tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                if hasattr(tts.backbone, "merge_and_unload"):
                    tts.backbone = tts.backbone.merge_and_unload()
                    tts._lora_loaded = False
            using_lmdeploy = False

        current_backbone = backbone_choice
        current_codec = codec_choice
        model_loaded = True

        success_msg = get_model_status_message()
        if lmdeploy_error_reason:
            success_msg += f"\n\n⚠️ **Cảnh báo:** Không thể kích hoạt LMDeploy: {lmdeploy_error_reason}"

        try: voices = tts.list_preset_voices()
        except: voices = []

        if voices:
            default_v = tts._default_voice
            voice_values = [v[1] for v in voices] if isinstance(voices[0], tuple) else voices
            if not default_v and voice_values: default_v = voice_values[0]
            if isinstance(voices[0], tuple): voices.sort(key=lambda x: str(x[0]))
            else: voices.sort()
            voice_update = gr.update(choices=voices, value=default_v, interactive=True)
            tab_p = tab_c = gr.update(visible=True)
            tab_sel = gr.update(selected="preset_mode")
            mode_state = "preset_mode"
        else:
            msg = "⚠️ Không tìm thấy file voices.json. Vui lòng dùng Tab Voice Cloning."
            voice_update = gr.update(choices=[msg], value=msg, interactive=False)
            tab_p = tab_c = gr.update(visible=True)
            tab_sel = gr.update(selected="preset_mode")
            mode_state = "preset_mode"

        yield (
            success_msg,
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False),
            voice_update, tab_p, tab_c, tab_sel, mode_state,
        )

    except Exception as e:
        model_loaded = False
        using_lmdeploy = False
        yield (
            f"❌ Lỗi khi tải model: {str(e)}",
            gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )


def synthesize_speech(
    text: str, voice_choice: str, custom_audio, custom_text: str,
    mode_tab: str, generation_mode: str, use_batch: bool,
    max_batch_size_run: int, temperature: float, max_chars_chunk: int,
):
    global tts, model_loaded, using_lmdeploy
    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return
    if not text or not text.strip():
        yield None, "⚠️ Vui lòng nhập văn bản!"
        return

    yield None, "📄 Đang xử lý Reference..."
    try:
        ref_codes, ref_text_raw = resolve_reference_voice(tts, mode_tab, voice_choice, custom_audio, custom_text)
        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()
    except Exception as e:
        yield None, f"❌ Lỗi xử lý Reference Audio: {str(e)}"
        return

    if generation_mode == "Standard (Một lần)":
        normalized_text = _text_normalizer.normalize(text.strip())
        text_chunks = split_text_into_chunks(normalized_text, max_chars=max_chars_chunk)
        total_chunks = len(text_chunks)
        yield None, f"🚀 Bắt đầu tổng hợp ({total_chunks} đoạn)..."

        all_wavs = []
        sr = 24000
        start_time = time.time()
        try:
            if use_batch and using_lmdeploy and hasattr(tts, "infer_batch") and total_chunks > 1:
                chunk_wavs = tts.infer_batch(
                    text_chunks, ref_codes=ref_codes, ref_text=ref_text_raw,
                    max_batch_size=max_batch_size_run, temperature=temperature, skip_normalize=True
                )
                all_wavs = [w for w in chunk_wavs if w is not None and len(w) > 0]
            else:
                for i, chunk in enumerate(text_chunks):
                    yield None, f"⏳ Đang xử lý đoạn {i + 1}/{total_chunks}..."
                    chunk_wav = tts.infer(
                        chunk, ref_codes=ref_codes, ref_text=ref_text_raw,
                        temperature=temperature, max_chars=max_chars_chunk, skip_normalize=True
                    )
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_wavs.append(chunk_wav)

            if not all_wavs:
                yield None, "❌ Không sinh được audio nào."
                return

            yield None, "💾 Đang ghép file và lưu..."
            final_wav = join_audio_chunks(all_wavs, sr=sr, silence_p=0.15)
            output_path = wrap_tempfile_sf_write(final_wav, sr)
            process_time = time.time() - start_time
            yield output_path, f"✅ Hoàn tất! ({process_time:.2f}s)"
            cleanup_gpu_memory()
        except Exception as e:
            yield None, f"❌ Lỗi: {str(e)}"
            cleanup_gpu_memory()

    else: # Streaming
        sr = 24000
        crossfade_samples = int(sr * 0.03)
        audio_queue = queue.Queue(maxsize=100)
        end_event = threading.Event()
        error_event = threading.Event()
        error_msg = ""
        text_chunks = split_text_into_chunks(_text_normalizer.normalize(text.strip()), max_chars=max_chars_chunk)

        def producer():
            nonlocal error_msg
            try:
                prev_tail = None
                for chunk in text_chunks:
                    for audio_part in tts.infer_stream(chunk, ref_codes=ref_codes, ref_text=ref_text_raw, temperature=temperature, skip_normalize=True):
                        if audio_part is None or len(audio_part) == 0: continue
                        if prev_tail is not None:
                            overlap = min(len(prev_tail), len(audio_part), crossfade_samples)
                            if overlap > 0:
                                blended = audio_part[:overlap] * np.linspace(0, 1, overlap) + prev_tail[-overlap:] * np.linspace(1, 0, overlap)
                                processed = np.concatenate([prev_tail[:-overlap], blended, audio_part[overlap:]])
                            else: processed = np.concatenate([prev_tail, audio_part])
                            tail_size = min(crossfade_samples, len(processed))
                            prev_tail = processed[-tail_size:].copy()
                            audio_queue.put((sr, processed[:-tail_size]))
                        else:
                            prev_tail = audio_part[-min(crossfade_samples, len(audio_part)):].copy()
                            audio_queue.put((sr, audio_part[:-len(prev_tail)]))
                if prev_tail is not None: audio_queue.put((sr, prev_tail))
            except Exception as e:
                error_msg = str(e); error_event.set()
            finally: end_event.set(); audio_queue.put(None)

        threading.Thread(target=producer, daemon=True).start()
        full_audio = []
        while not (end_event.is_set() and audio_queue.empty()):
            try:
                item = audio_queue.get(timeout=0.1)
                if item is None: break
                full_audio.append(item[1])
                yield item, "🔊 Đang phát..."
            except queue.Empty:
                if error_event.is_set(): yield None, f"❌ Lỗi: {error_msg}"; return
        if full_audio:
            yield wrap_tempfile_sf_write(np.concatenate(full_audio), sr), "✅ Hoàn tất!"


with gr.Blocks(theme=UI_THEME, css=COMMON_CSS, title="VieNeu-TTS", head=FAVICON_HEAD) as demo:
    with gr.Column(elem_classes="container"):
        gr.HTML(HEADER_HTML)
        with gr.Group():
            with gr.Row():
                backbone_select = gr.Dropdown(list(BACKBONE_CONFIGS.keys()) + ["Custom Model"], value="VieNeu-TTS (GPU)", label="🦜 Backbone")
                codec_select = gr.Dropdown(list(CODEC_CONFIGS.keys()), value="NeuCodec (Distill)", label="🎵 Codec")
                device_choice = gr.Radio(get_available_devices(), value="Auto", label="🖥️ Device")
            with gr.Row(visible=False) as custom_model_group:
                custom_backbone_model_id = gr.Textbox(label="📦 Custom Model ID", scale=2)
                custom_backbone_hf_token = gr.Textbox(label="🔑 HF Token", type="password", scale=1)
                custom_backbone_base_model = gr.Dropdown([k for k in BACKBONE_CONFIGS.keys() if "gguf" not in k.lower()], label="🔗 Base Model", value="VieNeu-TTS-0.3B (GPU)", visible=False, scale=1)
            use_lmdeploy_cb = gr.Checkbox(value=True, label="🚀 Optimize with LMDeploy")
            btn_load = gr.Button("🔄 Tải Model", variant="primary")
            model_status = gr.Markdown("⏳ Chưa tải model.")

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(label="Văn bản", lines=4, value="Hà Nội, trái tim của Việt Nam...")
                with gr.Tabs() as tabs:
                    with gr.TabItem("👤 Preset", id="preset_mode") as tab_preset:
                        voice_select = gr.Dropdown(choices=[], label="Giọng mẫu")
                    with gr.TabItem("🦜 Voice Cloning", id="custom_mode") as tab_custom:
                        custom_audio = gr.Audio(label="Audio giọng mẫu", type="filepath")
                        cloning_warning_msg = gr.Markdown(visible=False)
                        custom_text = gr.Textbox(label="Nội dung audio mẫu")
                generation_mode = gr.Radio(["Standard (Một lần)", "Streaming (Phát trực tiếp)"], value="Standard (Một lần)", label="Chế độ sinh")
                use_batch = gr.Checkbox(value=True, label="⚡ Batch Processing")
                max_batch_size_run = gr.Slider(1, 16, 4, step=1, label="📊 Batch Size")
                with gr.Accordion("⚙️ Cài đặt nâng cao", open=False):
                    temperature_slider = gr.Slider(0.1, 1.5, 1.0, step=0.1, label="🌡️ Temperature")
                    max_chars_chunk_slider = gr.Slider(128, 512, 256, step=32, label="📝 Max Chars")
                current_mode_state = gr.State("preset_mode")
                with gr.Row():
                    btn_generate = gr.Button("🎵 Bắt đầu", variant="primary", scale=2, interactive=False)
                    btn_stop = gr.Button("⏹️ Dừng", variant="stop", scale=1, interactive=False)
            with gr.Column(scale=2):
                audio_output = gr.Audio(label="Kết quả", type="filepath", autoplay=True)
                status_output = gr.Textbox(label="Trạng thái", elem_classes="status-box")

        codec_select.change(on_codec_change, [codec_select, current_mode_state], [tab_custom, tabs, current_mode_state])
        tab_preset.select(lambda: "preset_mode", None, current_mode_state)
        tab_custom.select(lambda: "custom_mode", None, current_mode_state)
        custom_audio.change(validate_audio_duration, [custom_audio], [cloning_warning_msg])
        backbone_select.change(on_backbone_change, [backbone_select], [custom_model_group])
        custom_backbone_model_id.change(on_custom_id_change, [custom_backbone_model_id], [custom_backbone_base_model])
        btn_load.click(load_model, [backbone_select, codec_select, device_choice, use_lmdeploy_cb, custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token], [model_status, btn_generate, btn_load, btn_stop, voice_select, tab_preset, tab_custom, tabs, current_mode_state])
        gen_ev = btn_generate.click(synthesize_speech, [text_input, voice_select, custom_audio, custom_text, current_mode_state, generation_mode, use_batch, max_batch_size_run, temperature_slider, max_chars_chunk_slider], [audio_output, status_output])
        btn_generate.click(lambda: gr.update(interactive=True), None, btn_stop)
        gen_ev.then(lambda: gr.update(interactive=False), None, btn_stop)
        btn_stop.click(None, None, None, cancels=[gen_ev])
        demo.load(restore_ui_state, None, [model_status, btn_generate, btn_stop])

if __name__ == "__main__":
    demo.queue().launch(server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")), share=env_bool("GRADIO_SHARE", False))
