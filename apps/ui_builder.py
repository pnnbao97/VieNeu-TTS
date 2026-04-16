import gradio as gr
import os
import soundfile as sf
import apps.model_manager as model_manager

def build_ui(BACKBONE_CONFIGS, CODEC_CONFIGS, DEFAULT_TEXT_GPU, DEFAULT_TEXT_TURBO, get_available_devices, is_xpu: bool = False):
    # Favicon (Parrot Emoji)
    head_html = """
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🦜</text></svg>">
    """

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

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui'],
    ).set(
        button_primary_background_fill="linear-gradient(90deg, #6366f1 0%, #0ea5e9 100%)",
        button_primary_background_fill_hover="linear-gradient(90deg, #4f46e5 0%, #0284c7 100%)",
    )

    title = "VieNeu-TTS (XPU)" if is_xpu else "VieNeu-TTS"
    header_text = "VieNeu-TTS Studio (Intel XPU Edition)" if is_xpu else "VieNeu-TTS Studio"

    with gr.Blocks(theme=theme, css=css, title=title, head=head_html) as demo:
        with gr.Column(elem_classes="container"):
            gr.HTML(f"""
    <div class="header-box">
        <h1 class="header-title">
            <span class="header-icon">🦜</span>
            <span class="gradient-text">{header_text}</span>
        </h1>
        <div class="model-card-content">
            <div class="model-card-item">
                <strong>Models:</strong>
                <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS" target="_blank" class="model-card-link">VieNeu-TTS</a>
                <span>•</span>
                <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B" target="_blank" class="model-card-link">VieNeu-TTS-0.3B</a>
                <span>•</span>
                <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2-Turbo" target="_blank" class="model-card-link">VieNeu-TTS-v2 (Turbo)</a>
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

            with gr.Group():
                with gr.Row():
                    if is_xpu:
                        default_backbone = "VieNeu-TTS (GPU)"
                        default_codec = "NeuCodec (Distill)"
                        default_temp = 1.0
                        default_text = DEFAULT_TEXT_GPU
                        backbone_select = gr.Dropdown(list(BACKBONE_CONFIGS.keys()) + ["Custom Model"], value=default_backbone, label="🦜 Backbone")
                        codec_select = gr.Dropdown(list(CODEC_CONFIGS.keys()), value=default_codec, label="🎵 Codec")
                        device_choice = gr.Radio(get_available_devices(), value="XPU", label="🖥️ Device", interactive=False)
                    else:
                        if "VieNeu-TTS (GPU)" in BACKBONE_CONFIGS:
                            default_backbone = "VieNeu-TTS (GPU)"
                        elif "VieNeu-TTS-v2-Turbo (GPU)" in BACKBONE_CONFIGS:
                            default_backbone = "VieNeu-TTS-v2-Turbo (GPU)"
                        elif "VieNeu-TTS-v2-Turbo (CPU)" in BACKBONE_CONFIGS:
                            default_backbone = "VieNeu-TTS-v2-Turbo (CPU)"
                        else:
                            default_backbone = list(BACKBONE_CONFIGS.keys())[0]

                        if "Turbo" in default_backbone:
                            default_codec = "VieNeu-Codec"
                            default_temp = 0.4
                            default_text = DEFAULT_TEXT_TURBO
                        else:
                            default_codec = "NeuCodec (Distill)" if "NeuCodec (Distill)" in CODEC_CONFIGS else list(CODEC_CONFIGS.keys())[0]
                            default_temp = 0.7
                            default_text = DEFAULT_TEXT_GPU

                        backbone_select = gr.Dropdown(list(BACKBONE_CONFIGS.keys()) + ["Custom Model"], value=default_backbone, label="🦜 Backbone")
                        codec_select = gr.Dropdown(list(CODEC_CONFIGS.keys()), value=default_codec, label="🎵 Codec", interactive=False)
                        device_choice = gr.Radio(get_available_devices(), value="Auto", label="🖥️ Device")

                with gr.Row(visible=False) as custom_model_group:
                    custom_backbone_model_id = gr.Textbox(label="📦 Custom Model ID", placeholder="pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen", info="Nhập HuggingFace Repo ID hoặc đường dẫn local", scale=2)
                    custom_backbone_hf_token = gr.Textbox(label="🔑 HF Token (nếu private)", placeholder="Để trống nếu repo public", type="password", info="Token để truy cập repo private", scale=1)
                    base_model_choices = [k for k in BACKBONE_CONFIGS.keys() if "turbo" not in k.lower() and k != "Custom Model"]
                    custom_backbone_base_model = gr.Dropdown(base_model_choices, label="🔗 Base Model (cho LoRA)", value=base_model_choices[0] if base_model_choices else None, visible=False, info="Model gốc để merge với LoRA (GPU Only)", scale=1)

                if not is_xpu:
                    with gr.Row():
                        use_lmdeploy_cb = gr.Checkbox(value=True, label="🚀 Optimize with LMDeploy (Khuyên dùng cho NVIDIA GPU)", info="Tick nếu bạn dùng GPU để tăng tốc độ tổng hợp đáng kể.")
                else:
                    use_lmdeploy_cb = gr.State(False)

                gr.Markdown("💡 **Sử dụng Custom Model:** Chọn \"Custom Model\" để tải LoRA adapter hoặc bất kỳ model nào được finetune từ **VieNeu-TTS** hoặc **VieNeu-TTS-0.3B**.")

                if is_xpu:
                    gr.HTML("""
                    <div class="warning-banner">
                        <div class="warning-banner-title">⚡ Chế độ tối ưu hóa cho Intel Arc GPU (XPU)</div>
                        <div class="warning-banner-content">Ứng dụng đang chạy trên pytorch nightly tối ưu hóa riêng cho card đồ họa Intel (PyTorch XPU).<br>Lần tạo giọng nói <b>đầu tiên</b> sẽ lâu hơn 1 chút.</div>
                    </div>
                    """)
                else:
                    gr.HTML("""
                    <div class="warning-banner">
                        <div class="warning-banner-title">🦜 Gợi ý tối ưu hiệu năng</div>
                        <div class="warning-banner-grid">
                            <div class="warning-banner-item">
                                <strong>🐆 Hệ máy GPU</strong>
                                <div class="warning-banner-content">Để có độ chính xác cao nhất và giọng đọc tự nhiên nhất, hãy sử dụng <b>VieNeu-TTS (Mặc định - GPU)</b>. Chọn <b>VieNeu-TTS-0.3B (GPU)</b> để tăng tốc độ lên gấp 2 lần.</div>
                            </div>
                            <div class="warning-banner-item" style="background: #dcfce7; border-color: #86efac;">
                                <strong style="color: #15803d;">🚀 VieNeu-TTS-v2</strong>
                                <div class="warning-banner-content" style="color: #166534;">Phiên bản <b>VieNeu-TTS-v2</b> đang trong quá trình phát triển nhằm hỗ trợ <b>song ngữ Anh-Việt</b>. Phiên bản <b>v2 Turbo</b> được ra mắt trước nhằm mục đích thử nghiệm.</div>
                            </div>
                        </div>
                    </div>
                    """)

                btn_load = gr.Button("🔄 Tải Model", variant="primary")
                model_status = gr.Markdown("⏳ Chưa tải model.")

            with gr.Row(elem_classes="container"):
                with gr.Column(scale=3):
                    text_input = gr.Textbox(label="Văn bản", lines=8, value=default_text)
                    with gr.Tabs() as tabs:
                        with gr.TabItem("👤 Preset", id="preset_mode") as tab_preset:
                            voice_select = gr.Dropdown(choices=[], value=None, label="Giọng mẫu")
                        with gr.TabItem("🦜 Voice Cloning", id="custom_mode") as tab_custom:
                            with gr.Group(visible=True) as cloning_elements_group:
                                custom_audio = gr.Audio(label="Audio giọng mẫu (3-5 giây) (.wav)", type="filepath")
                                cloning_warning_msg = gr.Markdown(visible=False, elem_id="cloning-warning")
                                custom_text = gr.Textbox(label="Nội dung audio mẫu")
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

                    generation_mode = gr.Radio(["Standard (Một lần)"], value="Standard (Một lần)", label="Chế độ sinh")
                    with gr.Row():
                        if is_xpu:
                            use_batch = gr.Checkbox(value=True, label="⚡ Batch Processing", info="Xử lý nhiều đoạn cùng lúc. Nên luôn chọn bật để tăng tốc độ.")
                            max_batch_size_run = gr.Slider(minimum=1, maximum=256, value=128, step=1, label="📊 Batch Size (Generation)")
                        else:
                            use_batch = gr.Checkbox(value=True, label="⚡ Batch Processing", info="Xử lý nhiều đoạn cùng lúc (GPU + LMDeploy)")
                            max_batch_size_run = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="📊 Batch Size (Generation)")

                    with gr.Accordion("⚙️ Cài đặt nâng cao (Generation)", open=False):
                        with gr.Row():
                            temperature_slider = gr.Slider(minimum=0.1, maximum=1.5, value=default_temp, step=0.1, label="🌡️ Temperature")
                            if is_xpu:
                                max_chars_chunk_slider = gr.Slider(minimum=64, maximum=512, value=128, step=16, label="📝 Max Chars per Chunk")
                            else:
                                max_chars_chunk_slider = gr.Slider(minimum=128, maximum=512, value=256, step=32, label="📝 Max Chars per Chunk")

                    current_mode_state = gr.State("preset_mode")
                    with gr.Row():
                        btn_generate = gr.Button("🎵 Bắt đầu", variant="primary", scale=2, interactive=False)
                        btn_stop = gr.Button("⏹️ Dừng", variant="stop", scale=1, interactive=False)

                with gr.Column(scale=2):
                    audio_output = gr.Audio(label="Kết quả", type="filepath", autoplay=True)
                    status_output = gr.Textbox(label="Trạng thái", elem_classes="status-box", lines=2, max_lines=10, show_copy_button=True)
                    gr.Markdown("<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>🔒 Audio được đóng dấu bản quyền ẩn (Watermarker).</div>")

        return (demo, backbone_select, codec_select, device_choice, use_lmdeploy_cb,
                custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token,
                btn_load, model_status, voice_select, tab_preset, tab_custom, tabs,
                current_mode_state, text_input, custom_audio, custom_text, generation_mode,
                use_batch, max_batch_size_run, temperature_slider, max_chars_chunk_slider,
                btn_generate, btn_stop, audio_output, status_output, cloning_warning_msg,
                cloning_elements_group, custom_model_group)

def validate_audio_duration(audio_path):
    if not audio_path:
        return gr.update(visible=False)
    try:
        info = sf.info(audio_path)
        if info.duration > 5.1:
            return gr.update(value=f"⚠️ **Cảnh báo:** Audio mẫu dài {info.duration:.1f} giây. Lý tưởng là 3-5 giây.", visible=True)
    except Exception:
        pass
    return gr.update(visible=False)
