import gradio as gr
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
from .ui_shared import (
    BACKBONE_CONFIGS, CODEC_CONFIGS, cleanup_gpu_memory, on_codec_change,
    validate_audio_duration, on_backbone_change, on_custom_id_change,
    resolve_reference_voice, get_backbone_config, wrap_tempfile_sf_write,
    UI_THEME, COMMON_CSS, HEADER_HTML, FAVICON_HEAD, _text_normalizer
)

print("⏳ Đang khởi động VieNeu-TTS... Vui lòng chờ...")

tts = None
current_backbone = current_codec = None
model_loaded = using_lmdeploy = False

def get_available_devices():
    devices = ["Auto", "CPU"]
    if sys.platform == "darwin":
        if torch.backends.mps.is_available(): devices.append("MPS")
    elif torch.cuda.is_available(): devices.append("CUDA")
    return devices

def get_model_status_message():
    global model_loaded, tts, using_lmdeploy, current_backbone, current_codec
    if not model_loaded or tts is None: return "⏳ Chưa tải model."
    backend = "🚀 LMDeploy" if using_lmdeploy else "📦 Standard"
    return f"✅ Model đã tải thành công!\n\n🔧 Backend: {backend}\n🦜 Backbone: {current_backbone}\n🎵 Codec: {current_codec}"

def load_model(backbone_choice, codec_choice, device_choice, force_lmdeploy, custom_model_id="", custom_base_model="", custom_hf_token=""):
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    model_loaded = False
    yield "⏳ Đang tải model...", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    try:
        if tts is not None: tts = None; cleanup_gpu_memory()
        backbone_config, custom_loading, is_merged_lora = get_backbone_config(backbone_choice, custom_model_id, custom_base_model)
        codec_config = CODEC_CONFIGS[codec_choice]
        use_lmdeploy = force_lmdeploy and (torch.cuda.is_available() if device_choice in ["Auto", "CUDA"] else False) and "gguf" not in backbone_config["repo"].lower()

        if use_lmdeploy:
            target_repo = backbone_config["repo"]
            if custom_loading and is_merged_lora:
                safe_name = custom_model_id.strip().replace("/", "_").replace("\\", "_")
                cache_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "merged_models_cache", safe_name))
                target_repo = cache_dir
                if not os.path.exists(os.path.join(cache_dir, "vocab.json")):
                    yield "⏳ Merging LoRA...", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    temp_tts = VieNeuTTS(BACKBONE_CONFIGS[custom_base_model]["repo"], hf_token=custom_hf_token)
                    temp_tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                    if hasattr(temp_tts.backbone, "merge_and_unload"): temp_tts.backbone = temp_tts.backbone.merge_and_unload()
                    temp_tts.backbone.save_pretrained(cache_dir); temp_tts.tokenizer.save_pretrained(cache_dir)
                    import json
                    with open(os.path.join(cache_dir, "voices.json"), "w", encoding="utf-8") as f:
                        json.dump({"default_voice": temp_tts._default_voice, "presets": temp_tts._preset_voices}, f)
                    del temp_tts; cleanup_gpu_memory()
            tts = FastVieNeuTTS(backbone_repo=target_repo, codec_repo=codec_config["repo"], hf_token=custom_hf_token)
            using_lmdeploy = True
        else:
            backbone_device = "mps" if sys.platform == "darwin" and device_choice == "Auto" else device_choice.lower()
            if "gguf" in backbone_config["repo"].lower(): backbone_device = "gpu"
            tts = VieNeuTTS(backbone_repo=backbone_config["repo"], backbone_device=backbone_device, codec_repo=codec_config["repo"], hf_token=custom_hf_token)
            if is_merged_lora:
                tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                if hasattr(tts.backbone, "merge_and_unload"): tts.backbone = tts.backbone.merge_and_unload()
            using_lmdeploy = False

        current_backbone, current_codec, model_loaded = backbone_choice, codec_choice, True
        voices = tts.list_preset_voices() or []
        voice_update = gr.update(choices=voices, value=tts._default_voice if voices else None, interactive=True)
        yield get_model_status_message(), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), voice_update, gr.update(visible=True), gr.update(visible=True), gr.update(selected="preset_mode"), "preset_mode"
    except Exception as e:
        yield f"❌ Lỗi: {e}", gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

def synthesize_speech(text, voice_choice, custom_audio, custom_text, mode_tab, generation_mode, use_batch, max_batch_size, temperature, max_chars):
    global tts, model_loaded, using_lmdeploy
    if not model_loaded: yield None, "⚠️ Tải model trước!"; return
    try:
        ref_codes, ref_text = resolve_reference_voice(tts, mode_tab, voice_choice, custom_audio, custom_text)
        if isinstance(ref_codes, torch.Tensor): ref_codes = ref_codes.cpu().numpy()
    except Exception as e: yield None, f"❌ Lỗi: {e}"; return

    if generation_mode == "Standard (Một lần)":
        chunks = split_text_into_chunks(_text_normalizer.normalize(text.strip()), max_chars=max_chars)
        all_wavs = []
        try:
            if use_batch and using_lmdeploy and hasattr(tts, "infer_batch") and len(chunks) > 1:
                all_wavs = [w for w in tts.infer_batch(chunks, ref_codes=ref_codes, ref_text=ref_text, max_batch_size=max_batch_size, temperature=temperature, skip_normalize=True) if w is not None]
            else:
                for c in chunks:
                    w = tts.infer(c, ref_codes=ref_codes, ref_text=ref_text, temperature=temperature, skip_normalize=True)
                    if w is not None: all_wavs.append(w)
            if not all_wavs: yield None, "❌ Lỗi sinh audio"; return
            yield wrap_tempfile_sf_write(join_audio_chunks(all_wavs, sr=24000), 24000), "✅ Hoàn tất!"
        except Exception as e: yield None, f"❌ Lỗi: {e}"
        finally: cleanup_gpu_memory()
    else: # Streaming
        sr, q, end_ev, err_ev = 24000, queue.Queue(), threading.Event(), threading.Event()
        def producer():
            try:
                for c in split_text_into_chunks(_text_normalizer.normalize(text.strip()), max_chars=max_chars):
                    for p in tts.infer_stream(c, ref_codes=ref_codes, ref_text=ref_text, temperature=temperature, skip_normalize=True):
                        if p is not None: q.put((sr, p))
            except Exception as e: err_ev.set(); print(e)
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
            btn_load = gr.Button("🔄 Tải Model", variant="primary"); model_status = gr.Markdown("⏳ Chưa tải model.")
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(label="Văn bản", lines=4, value="Hà Nội, trái tim của Việt Nam...")
                with gr.Tabs() as tabs:
                    with gr.TabItem("👤 Preset", id="preset_mode") as tab_preset: voice_select = gr.Dropdown(choices=[], label="Giọng mẫu")
                    with gr.TabItem("🦜 Voice Cloning", id="custom_mode") as tab_custom:
                        custom_audio, custom_text = gr.Audio(label="Audio mẫu", type="filepath"), gr.Textbox(label="Nội dung audio mẫu")
                        cloning_warning_msg = gr.Markdown(visible=False)
                generation_mode = gr.Radio(["Standard (Một lần)", "Streaming (Phát trực tiếp)"], value="Standard (Một lần)", label="Chế độ sinh")
                use_batch, max_batch_size_run = gr.Checkbox(value=True, label="⚡ Batch Processing"), gr.Slider(1, 16, 4, step=1, label="📊 Batch Size")
                with gr.Accordion("⚙️ Nâng cao", open=False):
                    temperature_slider, max_chars_chunk_slider = gr.Slider(0.1, 1.5, 1.0, step=0.1, label="🌡️ Temp"), gr.Slider(128, 512, 256, step=32, label="📝 Max Chars")
                mode_state = gr.State("preset_mode")
                with gr.Row(): btn_generate, btn_stop = gr.Button("🎵 Bắt đầu", variant="primary", scale=2, interactive=False), gr.Button("⏹️ Dừng", variant="stop", scale=1, interactive=False)
            with gr.Column(scale=2): audio_output, status_output = gr.Audio(label="Kết quả", type="filepath", autoplay=True), gr.Textbox(label="Trạng thái", elem_classes="status-box")
        codec_select.change(on_codec_change, [codec_select, mode_state], [tab_custom, tabs, mode_state])
        tab_preset.select(lambda: "preset_mode", None, mode_state); tab_custom.select(lambda: "custom_mode", None, mode_state)
        custom_audio.change(validate_audio_duration, [custom_audio], [cloning_warning_msg])
        backbone_select.change(on_backbone_change, [backbone_select], [custom_model_group])
        custom_backbone_model_id.change(on_custom_id_change, [custom_backbone_model_id], [custom_backbone_base_model])
        btn_load.click(load_model, [backbone_select, codec_select, device_choice, use_lmdeploy_cb, custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token], [model_status, btn_generate, btn_load, btn_stop, voice_select, tab_preset, tab_custom, tabs, mode_state])
        gen_ev = btn_generate.click(synthesize_speech, [text_input, voice_select, custom_audio, custom_text, mode_state, generation_mode, use_batch, max_batch_size_run, temperature_slider, max_chars_chunk_slider], [audio_output, status_output])
        btn_generate.click(lambda: gr.update(interactive=True), None, btn_stop)
        gen_ev.then(lambda: gr.update(interactive=False), None, btn_stop); btn_stop.click(None, None, None, cancels=[gen_ev])
        demo.load(lambda: (get_model_status_message(), gr.update(interactive=model_loaded), gr.update(interactive=False)), None, [model_status, btn_generate, btn_stop])

if __name__ == "__main__":
    demo.queue().launch(server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")), share=env_bool("GRADIO_SHARE", False))
