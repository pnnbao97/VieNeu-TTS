import gradio as gr
import os
import sys
import threading
import torch
import subprocess
from sea_g2p import Normalizer
import apps.model_manager as model_manager
import apps.inference_runner as inference_runner
import apps.ui_builder as ui_builder
from vieneu_utils.core_utils import env_bool

# --- Add XPU dll path ---
intel_dll_path = os.path.join(sys.prefix, 'Library', 'bin')
if os.path.exists(intel_dll_path):
    os.environ['PATH'] = intel_dll_path + os.pathsep + os.environ['PATH']

# Global lock for thread safety
model_lock = threading.Lock()

# Normalizer (module-level singleton)
_text_normalizer = Normalizer()

DEFAULT_TEXT_GPU = "Hà Nội, trái tim của Việt Nam, là một thành phố ngàn năm văn hiến với bề dày lịch sử và văn hóa độc đáo. Bước chân trên những con phố cổ kính quanh Hồ Hoàn Kiếm, du khách như được du hành ngược thời gian, chiêm ngưỡng kiến trúc Pháp cổ điển hòa quyện với nét kiến trúc truyền thống Việt Nam. Mỗi con phố trong khu phố cổ mang một tên gọi đặc trưng, phản ánh nghề thủ công truyền thống từng thịnh hành nơi đây như phố Hàng Bạc, Hàng Đào, Hàng Mã. Ẩm thực Hà Nội cũng là một điểm nhấn đặc biệt, từ tô phở nóng hổi buổi sáng, bún chả thơm lừng trưa hè, đến chè Thái ngọt ngào chiều thu. Những món ăn dân dã này đã trở thành biểu tượng của văn hóa ẩm thực Việt, được cả thế giới yêu mến. Người Hà Nội nổi tiếng with tính cách hiền hòa, lịch thiệp nhưng cũng rất cầu toàn trong từng chi tiết nhỏ, từ cách pha trà sen cho đến cách chọn hoa sen tây để thưởng trà."

def safe_load_model(*args):
    with model_lock:
        yield from model_manager.load_model(*args, is_xpu=True)

def safe_synthesize_speech(*args):
    with model_lock:
        yield from inference_runner.synthesize_speech(*args, _text_normalizer=_text_normalizer, is_xpu=True)

def main():
    try:
        if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
            print("⚠️ Không tìm thấy thiết bị Intel XPU (Intel Arc GPU).")
            print("🔄 Đang tự động chuyển hướng sang phiên bản CPU/CUDA (gradio_app.py)...")
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "gradio_main.py")] + sys.argv[1:])
            return
    except ImportError:
        pass

    print("⏳ Đang khởi động VieNeu-TTS (Phiên bản tối ưu cho Intel XPU)...")

    (demo, backbone_select, codec_select, device_choice, _,
     custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token,
     btn_load, model_status, voice_select, tab_preset, tab_custom, tabs,
     current_mode_state, text_input, custom_audio, custom_text, generation_mode,
     use_batch, max_batch_size_run, temperature_slider, max_chars_chunk_slider,
     btn_generate, btn_stop, audio_output, status_output, cloning_warning_msg,
     cloning_elements_group, custom_model_group) = ui_builder.build_ui(
         model_manager.BACKBONE_CONFIGS, model_manager.CODEC_CONFIGS,
         DEFAULT_TEXT_GPU, "", model_manager.get_available_devices, is_xpu=True
     )

    with demo:
        def on_codec_change(codec: str, current_mode: str):
            is_onnx = "onnx" in codec.lower()
            if is_onnx and current_mode == "custom_mode":
                return gr.update(visible=False), gr.update(selected="preset_mode"), "preset_mode"
            return gr.update(visible=not is_onnx), gr.update(), current_mode
        
        codec_select.change(on_codec_change, inputs=[codec_select, current_mode_state], outputs=[tab_custom, tabs, current_mode_state])
        tab_preset.select(lambda: "preset_mode", outputs=current_mode_state)
        tab_custom.select(lambda: "custom_mode", outputs=current_mode_state)
        custom_audio.change(ui_builder.validate_audio_duration, inputs=[custom_audio], outputs=[cloning_warning_msg])
        
        backbone_select.change(lambda choice: gr.update(visible=choice == "Custom Model"), inputs=[backbone_select], outputs=[custom_model_group])
        
        def on_custom_id_change(model_id):
            if model_id and "lora" in model_id.lower():
                base_model = "VieNeu-TTS-0.3B (GPU)" if "0.3" in model_id else "VieNeu-TTS (GPU)"
                return gr.update(visible=True, value=base_model), gr.update(), gr.update()
            return gr.update(visible=False), gr.update(), gr.update()
            
        custom_backbone_model_id.change(on_custom_id_change, inputs=[custom_backbone_model_id], outputs=[custom_backbone_base_model, custom_audio, custom_text])

        btn_load.click(
            fn=safe_load_model,
            inputs=[backbone_select, codec_select, device_choice, 
                    custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
            outputs=[model_status, btn_generate, btn_load, btn_stop, voice_select, tab_preset, tab_custom, tabs, current_mode_state]
        )
        
        generate_event = btn_generate.click(
            fn=safe_synthesize_speech,
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

        demo.load(fn=lambda: model_manager.restore_ui_state(is_xpu=True), outputs=[model_status, btn_generate, btn_stop])

    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = env_bool("GRADIO_SHARE", default=os.getenv("COLAB_RELEASE_TAG") is not None)
    if server_name == "0.0.0.0" and os.getenv("GRADIO_SHARE") is None:
        share = False

    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)

if __name__ == "__main__":
    main()
