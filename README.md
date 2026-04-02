# 🦜 VieNeu-TTS

[![Awesome](https://img.shields.io/badge/Awesome-NLP-green?logo=github)](https://github.com/keon/awesome-nlp)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yJt8kzjzWZ)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b9PO-lcGZX9pEkEwQmu8MfhSnjxKrALW?usp=sharing)
[![Hugging Face v2 Turbo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v2%20Turbo-blue)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2-Turbo)
[![Hugging Face VieNeu-TTS](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v1-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)

<img width="1087" height="710" alt="image" src="https://github.com/user-attachments/assets/5534b5db-f30b-4d27-8a35-80f1cf6e5d4d" />

**VieNeu-TTS** is an advanced on-device Vietnamese Text-to-Speech (TTS) model with **instant voice cloning** and **English-Vietnamese bilingual** support.

> [!IMPORTANT]
> **🚀 VieNeu-TTS-v2 Turbo:** Optimized for edge devices and extremely fast inference (CPU & Low-end devices).  
> *Note: Quality is lower than the Standard VieNeu-TTS and may struggle with very short segments (< 5 words).*  
> Version **VieNeu-TTS-v2 (Non-Turbo)** is coming soon!

## ✨ Key Features
- **Bilingual (English-Vietnamese)**: Smooth and natural transitions between languages powered by [sea-g2p](https://github.com/pnnbao97/sea-g2p).
- **Instant Voice Cloning**: Clone any voice with just **3-5 seconds** of reference audio (**Turbo v2** & GPU modes).
- **Ultra-Fast Turbo Mode**: Optimized for both **CPU (GGUF)** and **GPU (LMDeploy)**, offering the fastest inference in the VieNeu family.
- **AI Identification**: Built-in audio watermarking for responsible AI content creation.
- **Production-Ready**: High-quality 24 kHz waveform generation, fully offline.

[<img width="600" height="595" alt="VieNeu-TTS Demo" src="https://github.com/user-attachments/assets/021f6671-2d7f-4635-91fb-88b2ab0ddbcd" />](https://github.com/user-attachments/assets/021f6671-2d7f-4635-91fb-88b2ab0ddbcd)

## 📌 Table of Contents

1. [🦜 Installation & Web UI](#installation)
2. [📦 Using the Python SDK](#sdk)
3. [🐳 Docker & Remote Server](#docker-remote)
4. [🔬 Model Overview](#backbones)
5. [🚀 Roadmap](#roadmap)
6. [🤝 Support & Contact](#support)
7. [📑 Citation](#citation)

---

## 🦜 1. Installation & Web UI <a name="installation"></a>

### Setup with `uv` (Recommended)
`uv` is the fastest way to manage dependencies. 
```bash
# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/pnnbao97/VieNeu-TTS.git
   cd VieNeu-TTS
   ```

2. **Install Dependencies:**
   - **Option 1: Minimal (Turbo/CPU)** - Fast & Lightweight
     ```bash
     uv sync
     ```
   - **Option 2: Full (GPU/Standard)** - High Quality & Cloning
     ```bash
     uv sync --group gpu
     ```

3. **Start the Web UI:**
   ```bash
   uv run vieneu-web
   ```
   Access the UI at `http://127.0.0.1:7860`. The **Turbo v2** model is selected by default for immediate use.

---

## 📦 2. Using the Python SDK (vieneu) <a name="sdk"></a>

The `vieneu` SDK now defaults to **Turbo mode** for maximum compatibility.

### Quick Start
```bash
# Minimal installation (Builds llama-cpp from source - may take a while)
pip install vieneu

# Optional: For Windows users (CPU pre-built)
pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/

# Optional: For macOS users (ARM64/Apple Silicon - Enables Metal GPU acceleration)
pip install vieneu --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal/
```

```python
from vieneu import Vieneu

# Initialize in Turbo mode (Default - Minimal dependencies)
tts = Vieneu()

# 1. Simple synthesis (uses default Southern Male voice 'Xuân Vĩnh')
text = "Hệ thống điện chủ yếu sử dụng alternating current because it is more efficient."
audio = tts.infer(text=text)

# Save to file
tts.save(audio, "output_Xuân Vĩnh.wav")
print("💾 Saved to output_Xuân Vĩnh.wav")

# 2. Using a specific Preset Voice
voices = tts.list_preset_voices()
for desc, voice_id in voices:
    print(f"Voice: {desc} (ID: {voice_id})")

my_voice_id = voices[1][1] if len(voices) > 1 else voices[0][1] # Giọng Phạm Tuyên
voice_data = tts.get_preset_voice(my_voice_id)

audio_custom = tts.infer(text="Tôi đang nói bằng giọng của Bác sĩ Tuyên.", voice=voice_data)

# 3. Save to file
tts.save(audio_custom, "output_Phạm Tuyên.wav")
print("💾 Saved to output_Phạm Tuyên.wav")
```

### 🦜 Zero-shot Voice Cloning (SDK) <a name="cloning"></a>

Clone any voice with only **3-5 seconds** of audio using the local Turbo engine:

```python
from vieneu import Vieneu

tts = Vieneu() # Defaults to Turbo mode

# 1. Encode the reference audio
# Supported formats: .wav, .mp3, .flac (5-10 seconds recommended)
my_voice = tts.encode_reference("examples/audio_ref/example.wav")

# 2. Synthesize with the cloned voice
# No reference text required for Turbo v2!
audio = tts.infer(
    text="Đây là giọng nói được clone trực tiếp bằng SDK của VieNeu-TTS.", 
    voice=my_voice  # accepts numpy array from encode_reference() or preset dict from get_preset_voice()
)

tts.save(audio, "cloned_voice.wav")
```

---

## 🐳 3. Docker & Remote Server <a name="docker-remote"></a>

Deploy VieNeu-TTS as a high-performance API Server (powered by LMDeploy) with a single command.

### 1. Run with Docker (Recommended)

**Requirement**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is required for GPU support.

**Start the Server with a Public Tunnel (No port forwarding needed):**
```bash
docker run --gpus all -p 23333:23333 pnnbao/vieneu-tts:serve --tunnel
```

*   **Default**: The server loads the `VieNeu-TTS` model for maximum quality.
*   **Tunneling**: The Docker image includes a built-in `bore` tunnel. Check the container logs to find your public address (e.g., `bore.pub:31631`).

### 2. Using the SDK (Remote Mode)

Once the server is running, you can connect from anywhere (Colab, Web Apps, etc.) without loading heavy models locally:

```python
from vieneu import Vieneu
import os

# Configuration
REMOTE_API_BASE = 'http://your-server-ip:23333/v1'  # Or bore tunnel URL
REMOTE_MODEL_ID = "pnnbao-ump/VieNeu-TTS"

# Initialization (LIGHTWEIGHT - only loads small codec locally)
tts = Vieneu(mode='remote', api_base=REMOTE_API_BASE, model_name=REMOTE_MODEL_ID)
os.makedirs("outputs", exist_ok=True)

# List remote voices
available_voices = tts.list_preset_voices()
for desc, name in available_voices:
    print(f"   - {desc} (ID: {name})")

# Use specific voice (dynamically select second voice)
if available_voices:
    _, my_voice_id = available_voices[1]
    voice_data = tts.get_preset_voice(my_voice_id)
    audio_spec = tts.infer(text="Chào bạn, tôi đang nói bằng giọng của bác sĩ Tuyên.", voice=voice_data)
    tts.save(audio_spec, f"outputs/remote_{my_voice_id}.wav")
    print(f"💾 Saved synthesis to: outputs/remote_{my_voice_id}.wav")

# Standard synthesis (uses default voice)
text_input = "Chế độ remote giúp tích hợp VieNeu vào ứng dụng Web hoặc App cực nhanh mà không cần GPU tại máy khách."
audio = tts.infer(text=text_input)
tts.save(audio, "outputs/remote_output.wav")
print("💾 Saved remote synthesis to: outputs/remote_output.wav")

# Zero-shot voice cloning (encodes audio locally, sends codes to server)
if os.path.exists("examples/audio_ref/example_ngoc_huyen.wav"):
    cloned_audio = tts.infer(
        text="Đây là giọng nói được clone và xử lý thông qua VieNeu Server.",
        ref_audio="examples/audio_ref/example_ngoc_huyen.wav",
        ref_text="Tác phẩm dự thi bảo đảm tính khoa học, tính đảng, tính chiến đấu, tính định hướng."
    )
    tts.save(cloned_audio, "outputs/remote_cloned_output.wav")
    print("💾 Saved remote cloned voice to: outputs/remote_cloned_output.wav")
```
*For full implementation details, see: [examples/main_remote.py](examples/main_remote.py)*

### Voice Preset Specification (v1.0)
VieNeu-TTS uses the official `vieneu.voice.presets` specification to define reusable voice assets. Only `voices.json` files following this spec are guaranteed to be compatible with VieNeu-TTS SDK ≥ v1.x.

### 3. Advanced Configuration

Customize the server to run specific versions or your own fine-tuned models.

**Run the 0.3B Model (Faster):**
```bash
docker run --gpus all pnnbao/vieneu-tts:serve --model pnnbao-ump/VieNeu-TTS-0.3B --tunnel
```

**Serve a Local Fine-tuned Model:**
If you have merged a LoRA adapter, mount your output directory to the container:
```bash
# Linux / macOS
docker run --gpus all \
  -v $(pwd)/finetune/output:/workspace/models \
  pnnbao/vieneu-tts:serve \
  --model /workspace/models/merged_model --tunnel
```

---

## 🔬 4. Model Overview <a name="backbones"></a>

| Model | Format | Device | Bilingual | Cloning | Speed |
|---|---|---|---|---|---|
| **VieNeu-v2-Turbo** | GGUF/ONNX | **CPU/Edge** | ✅ | ✅ Yes | **Extreme** (Fastest) |
| **VieNeu-TTS-v2** | PyTorch | GPU | ✅ | ✅ Yes | **Standard** (Coming soon) |
| **VieNeu-TTS 0.3B** | PyTorch | GPU/CPU | ❌ | ✅ Yes | **Very Fast** |
| **VieNeu-TTS** | PyTorch | GPU/CPU | ❌ | ✅ Yes | **Standard** |

> [!TIP]
> Use **Turbo v2** for AI assistants, chatbots, and real-time edge applications where speed is critical. Note: It may have stability issues with very short phrases (< 5 words).
> Use **GPU/Standard** (VieNeu-TTS v1/v2) for maximum audio quality and high-fidelity voice cloning.

---

## 🚀 5. Roadmap <a name="roadmap"></a>

- [x] **VieNeu-TTS-v2 Turbo**: English-Vietnamese code-switching support.
- [x] **VieNeu-Codec**: Optimized neural codec for Vietnamese (ONNX).
- [ ] **VieNeu-TTS-v2 (Non-Turbo)**: Full high-fidelity bilingual architecture with instant **Voice Cloning** and **LMDeploy** GPU acceleration support.
- [x] **Turbo Voice Cloning**: Bringing instant cloning to the lightweight Turbo engine.
- [ ] **Mobile SDK**: Official support for Android/iOS deployment.

---

## 🤝 6. Support & Contact <a name="support"></a>

- **Hugging Face:** [pnnbao-ump](https://huggingface.co/pnnbao-ump)
- **Discord:** [Join our community](https://discord.gg/yJt8kzjzWZ)
- **Facebook:** [Pham Nguyen Ngoc Bao](https://www.facebook.com/pnnbao97)
- **License:** Apache 2.0 (Free to use).

---
## 📑 7. Citation <a name="citation"></a>

```bibtex
@misc{vieneutts2026,
  title        = {VieNeu-TTS: Vietnamese Text-to-Speech with Instant Voice Cloning},
  author       = {Pham Nguyen Ngoc Bao},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/pnnbao-ump/VieNeu-TTS}}
}
```

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=pnnbao97/VieNeu-TTS&type=Date)](https://star-history.com/#pnnbao97/VieNeu-TTS&Date)

---

## 🤝 Contributors

Thanks to all the amazing people who have contributed to this project!

<a href="https://github.com/pnnbao97/VieNeu-TTS/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pnnbao97/VieNeu-TTS" />
</a>

---

## 🙏 Acknowledgements

This project uses [neucodec](https://huggingface.co/neuphonic/neucodec) for audio decoding and [sea-g2p](https://github.com/pnnbao97/sea-g2p) for text normalization and phonemization.

**Made with ❤️ for the Vietnamese TTS community**
