import gradio as gr
print("⏳ Đang khởi động... Vui lòng chờ...")
from app_config import (
    BACKBONE_CONFIGS,
    CODEC_CONFIGS,
    MAX_CHARS_PER_CHUNK,
    MAX_MULTI_LINES,
    MAX_TOTAL_CHARS_STREAMING,
    SPEAKER_MODE_LABELS,
)
from services.tts_service import get_available_devices, load_model, synthesize_router
from ui.app_assets import load_css, load_header_html, load_multi_voice_head
from ui.app_handlers import set_mode_custom, set_mode_multi, set_mode_preset, update_info
from ui.app_theme import build_theme
from ui.multi_voice_utils import (
    add_multi_line_with_order,
    clear_multi_lines_with_text,
    get_voice_options,
    remove_multi_line_at,
    update_multi_mode_visibility,
    update_multi_voice_dropdowns,
    update_voice_dropdown,
)
print("⏳ Đang khởi động VieNeu-TTS...")

def create_demo() -> gr.Blocks:
    theme = build_theme()
    css = load_css()
    multi_voice_head = load_multi_voice_head()
    header_html = load_header_html()

    with gr.Blocks(theme=theme, css=css, head=multi_voice_head, title="VieNeu-TTS") as demo:
        with gr.Column(elem_classes="container"):
            gr.HTML(header_html)

            # --- CONFIGURATION ---
            with gr.Group():
                with gr.Row():
                    backbone_select = gr.Dropdown(list(BACKBONE_CONFIGS.keys()), value="VieNeu-TTS (GPU)", label="🦜 Backbone")
                    codec_select = gr.Dropdown(list(CODEC_CONFIGS.keys()), value="NeuCodec (Standard)", label="🎵 Codec")
                    device_choice = gr.Radio(get_available_devices(), value="Auto", label="🖥️ Device")

                with gr.Row():
                    enable_triton = gr.Checkbox(value=True, label="⚡ Enable Triton Compilation")
                    max_batch_size = gr.Slider(
                        minimum=1, 
                        maximum=16, 
                        value=8, 
                        step=1, 
                        label="📊 Max Batch Size",
                        info="Giảm nếu gặp lỗi OOM. 4-6 cho GPU 8GB, 8-12 cho GPU 16GB+"
                    )

                gr.Markdown(
                    "⚠️ **Lưu ý:** Nếu máy bạn chỉ có CPU vui lòng chọn phiên bản GGUF (Q4/Q8) để có tốc độ nhanh nhất.\n\n"
                    "💡 **Max Batch Size:** Số lượng đoạn văn bản được xử lý cùng lúc. "
                    "Giá trị cao = nhanh hơn nhưng tốn VRAM hơn. Giảm xuống nếu gặp lỗi \"Out of Memory\"."
                )

                btn_load = gr.Button("🔄 Tải Model", variant="primary")
                model_status = gr.Markdown("⏳ Chưa tải model.")

            with gr.Row(elem_classes="container"):
                # --- INPUT ---
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label=f"Văn bản (Streaming hỗ trợ tới {MAX_TOTAL_CHARS_STREAMING} ký tự, chia chunk {MAX_CHARS_PER_CHUNK} ký tự)",
                        lines=4,
                        value="Hà Nội, trái tim của Việt Nam, là một thành phố ngàn năm văn hiến với bề dày lịch sử và văn hóa độc đáo. Bước chân trên những con phố cổ kính quanh Hồ Hoàn Kiếm, du khách như được du hành ngược thời gian, chiêm ngưỡng kiến trúc Pháp cổ điển hòa quyện với nét kiến trúc truyền thống Việt Nam. Mỗi con phố trong khu phố cổ mang một tên gọi đặc trưng, phản ánh nghề thủ công truyền thống từng thịnh hành nơi đây như phố Hàng Bạc, Hàng Đào, Hàng Mã. Ẩm thực Hà Nội cũng là một điểm nhấn đặc biệt, từ tô phở nóng hổi buổi sáng, bún chả thơm lừng trưa hè, đến chè Thái ngọt ngào chiều thu. Những món ăn dân dã này đã trở thành biểu tượng của văn hóa ẩm thực Việt, được cả thế giới yêu mến. Người Hà Nội nổi tiếng với tính cách hiền hòa, lịch thiệp nhưng cũng rất cầu toàn trong từng chi tiết nhỏ, từ cách pha trà sen cho đến cách chọn hoa sen tây để thưởng trà.",
                        elem_id="single-text-input",
                    )

                    with gr.Tabs() as tabs:
                        with gr.TabItem("👤 Preset", id="preset_mode") as tab_preset:
                            initial_voices = get_voice_options("VieNeu-TTS (GPU)")
                            default_voice = initial_voices[0] if initial_voices else None
                            voice_select = gr.Dropdown(initial_voices, value=default_voice, label="Giọng mẫu")
                        
                        with gr.TabItem("🦜 Voice Cloning", id="custom_mode") as tab_custom:
                            custom_audio = gr.Audio(label="Audio giọng mẫu (10-15 giây) (.wav)", type="filepath")
                            custom_text = gr.Textbox(label="Nội dung audio mẫu - vui lòng gõ đúng nội dung của audio mẫu - kể cả dấu câu vì model rất nhạy cảm với dấu câu (.,?!)")

                        with gr.TabItem("Hội thoại", id="multi_mode", elem_id="tab-multi") as tab_multi:
                            multi_initial_voices = get_voice_options("VieNeu-TTS (GPU)")
                            multi_default_voice = multi_initial_voices[0] if multi_initial_voices else None
                            multi_line_count = gr.State(1)
                            multi_line_order = gr.Textbox(value="1", visible=False, elem_id="multi-line-order")
                            with gr.Row(elem_classes="multi-title-row"):
                                gr.Markdown("### Hội thoại đa giọng", elem_classes="multi-title")
                                multi_add_line = gr.Button("+", variant="secondary", size="sm", elem_classes="icon-button", elem_id="multi-add-line")
                                multi_expand_all = gr.Button("All", variant="secondary", size="sm", elem_classes="icon-button", elem_id="multi-expand-all")
                                multi_clear_lines = gr.Button("↺", variant="secondary", size="sm", elem_classes="icon-button icon-danger", elem_id="multi-clear-lines")


                            multi_line_groups = []
                            multi_line_modes = []
                            multi_line_voices = []
                            multi_line_custom_audios = []
                            multi_line_custom_texts = []
                            multi_line_texts = []
                            multi_line_delete_buttons = []

                            with gr.Column(elem_id="multi-lines-container"):
                                for i in range(MAX_MULTI_LINES):
                                    with gr.Group(
                                        visible=(i == 0),
                                        elem_classes="multi-line-card",
                                        elem_id=f"multi-line-{i+1}",
                                    ) as line_group:
                                        with gr.Row(elem_classes="multi-line-header"):
                                            gr.Markdown(f"**Người nói {i+1}**", elem_classes="multi-line-title")
                                            gr.HTML("<div class='drag-handle' title='Kéo để sắp xếp'>⠿</div>")
                                            line_delete = gr.Button("X", variant="secondary", size="sm", elem_classes="icon-button remove-button icon-danger")
                                        with gr.Row():
                                            line_mode = gr.Radio(
                                                SPEAKER_MODE_LABELS,
                                                value="Giọng mẫu",
                                                label="Chế độ giọng"
                                            )
                                            line_voice = gr.Dropdown(
                                                multi_initial_voices,
                                                value=multi_default_voice,
                                                label="Giọng mẫu"
                                            )
                                        line_custom_audio = gr.Audio(
                                            label="Audio giọng tùy chỉnh (.wav)",
                                            type="filepath",
                                            visible=False
                                        )
                                        line_custom_text = gr.Textbox(
                                            label="Văn bản giọng tùy chỉnh",
                                            lines=2,
                                            visible=False
                                        )
                                        line_text = gr.Textbox(
                                            label="Nội dung",
                                            lines=3,
                                            placeholder="Nhập nội dung cho người nói này...",
                                            elem_classes="multi-line-text"
                                        )

                                    multi_line_groups.append(line_group)
                                    multi_line_modes.append(line_mode)
                                    multi_line_voices.append(line_voice)
                                    multi_line_custom_audios.append(line_custom_audio)
                                    multi_line_custom_texts.append(line_custom_text)
                                    multi_line_texts.append(line_text)
                                    multi_line_delete_buttons.append(line_delete)

                    generation_mode = gr.Radio(
                        ["Standard (Một lần)"],
                        value="Standard (Một lần)",
                        label="Chế độ sinh"
                    )
                    use_batch = gr.Checkbox(
                        value=True, 
                        label="⚡ Batch Processing",
                        info="Xử lý nhiều đoạn cùng lúc (chỉ áp dụng khi sử dụng GPU và đã cài đặt LMDeploy)"
                    )

                    # State to track current mode (replaces unreliable Textbox/Tabs input)
                    current_mode_state = gr.State("preset_mode")

                    btn_generate = gr.Button("🎵 Bắt đầu", variant="primary", size="lg", interactive=False)

                # --- OUTPUT ---
                with gr.Column(scale=2):
                    audio_output = gr.Audio(
                        label="Kết quả",
                        type="filepath",
                        autoplay=True
                    )
                    status_output = gr.Textbox(label="Trạng thái", elem_classes="status-box")

            multi_line_inputs = []
            for i in range(MAX_MULTI_LINES):
                multi_line_inputs.extend([
                    multi_line_modes[i],
                    multi_line_voices[i],
                    multi_line_custom_audios[i],
                    multi_line_custom_texts[i],
                    multi_line_texts[i],
                ])

            # --- EVENT HANDLERS ---
            backbone_select.change(update_info, backbone_select, model_status)
            backbone_select.change(update_voice_dropdown, [backbone_select, voice_select], voice_select)
            backbone_select.change(
                update_multi_voice_dropdowns,
                inputs=[backbone_select] + multi_line_voices,
                outputs=multi_line_voices,
            )

            # Bind tab events to update state
            tab_preset.select(set_mode_preset, outputs=[current_mode_state, text_input])
            tab_custom.select(set_mode_custom, outputs=[current_mode_state, text_input])
            tab_multi.select(set_mode_multi, outputs=[current_mode_state, text_input])

            for line_mode, line_voice, line_custom_audio, line_custom_text in zip(
                multi_line_modes,
                multi_line_voices,
                multi_line_custom_audios,
                multi_line_custom_texts,
            ):
                line_mode.change(
                    update_multi_mode_visibility,
                    inputs=line_mode,
                    outputs=[line_voice, line_custom_audio, line_custom_text],
                )

            multi_add_line.click(
                fn=lambda count: add_multi_line_with_order(count, MAX_MULTI_LINES),
                inputs=multi_line_count,
                outputs=[multi_line_count] + multi_line_groups + [multi_line_order],
            )
            multi_clear_lines.click(
                fn=lambda: clear_multi_lines_with_text(MAX_MULTI_LINES),
                outputs=[multi_line_count] + multi_line_groups + multi_line_texts + [multi_line_order],
            )

            for idx, line_delete in enumerate(multi_line_delete_buttons):
                line_delete.click(
                    fn=lambda line_count, backbone_choice, *line_inputs, idx=idx: remove_multi_line_at(
                        idx, line_count, backbone_choice, MAX_MULTI_LINES, *line_inputs
                    ),
                    inputs=[multi_line_count, backbone_select] + multi_line_inputs,
                    outputs=[multi_line_count] + multi_line_groups + multi_line_inputs + [multi_line_order],
                )

            btn_load.click(
                fn=load_model,
                inputs=[backbone_select, codec_select, device_choice, enable_triton, max_batch_size],
                outputs=[model_status, btn_generate, btn_load],
            )

            btn_generate.click(
                fn=synthesize_router,
                inputs=[
                    text_input,
                    voice_select,
                    custom_audio,
                    custom_text,
                    current_mode_state,
                    generation_mode,
                    use_batch,
                    multi_line_count,
                    multi_line_order,
                ] + multi_line_inputs,
                outputs=[audio_output, status_output],
            )

    return demo
