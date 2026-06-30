# 🦜 VieNeu-TTS

[![Awesome](https://img.shields.io/badge/Awesome-NLP-green?logo=github)](https://github.com/keon/awesome-nlp)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yJt8kzjzWZ)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b9PO-lcGZX9pEkEwQmu8MfhSnjxKrALW?usp=sharing)
[![Hugging Face VieNeu-TTS-v3-Turbo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v3--Turbo-red)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo)
[![Hugging Face VieNeu-TTS-v2](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v2-blue)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2)
[![Hugging Face VieNeu-TTS](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v1-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)

<img width="1087" height="710" alt="image" src="https://github.com/user-attachments/assets/5534b5db-f30b-4d27-8a35-80f1cf6e5d4d" />

**VieNeu-TTS-v2** is the next generation of on-device Vietnamese TTS, featuring **10,000+ hours** of bilingual training, **instant voice cloning**, and a dedicated **Podcast/Conversation** mode.

> [!NOTE]
> **🆕 VieNeu-TTS v3 Turbo (early access) is out for preview!**
> A brand-new architecture **designed and trained from scratch by Phạm Nguyễn Ngọc Bảo** (codec: [MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano); phonemizer: [sea-g2p](https://github.com/pnnbao97/sea-g2p)):
> - **48 kHz** high-fidelity audio (up from 24 kHz).
> - **Built-in default voices** via dedicated speaker tokens — stable, consistent, no reference clip needed.
> - **Emotion / non-verbal cues** *(experimental)*: drop `[cười]`, `[thở dài]`, `[hắng giọng]` straight into the text.
> - **Batched generation** (batch size up to 32), including a multi-speaker **Conversation** mode that batches the whole script regardless of speaker.
> - **Instant voice cloning** from 3–5s of audio.
>
> Try it in the Web UI (backbone **"VieNeu-TTS-v3-Turbo (Thử nghiệm)"**) or the SDK (`Vieneu(mode="v3turbo")`). The **full v3** release is coming in the next few weeks.

> [!IMPORTANT]
> **🚀 VieNeu-TTS-v2 is here!**
> The full high-fidelity bilingual architecture is now available with:
> - **10,000+ Hours of Data:** Unmatched naturalness in both English and Vietnamese.
> - **Podcast & Dialogue Mode:** Multi-speaker support with emotional nuances.
> - **Zero-shot Cloning:** Clone any voice in 3-5 seconds across all v2 variants.

## ✨ Key Features
- **10,000+ Hours Training**: Trained on a massive English-Vietnamese dataset for human-like prosody.
- **Bilingual (En-Vi) Code-switching**: Powered by [**sea-g2p**](https://github.com/pnnbao97/sea-g2p) for high-fidelity pronunciation and seamless transitions between Vietnamese and English.
- **Podcast & Conversation Mode**: Multi-speaker dialogue support with automatic character detection.
- **Instant Voice Cloning**: Clone any voice with just **3-5 seconds** of reference audio.
- **Ultra-Fast Performance**: Optimized for **GPU (LMDeploy)** and **CPU (GGUF/ONNX)**.
- **Production-Ready**: High-quality 24 kHz waveform generation, fully offline.

[<img width="600" height="595" alt="VieNeu-TTS Demo" src="https://github.com/user-attachments/assets/021f6671-2d7f-4635-91fb-88b2ab0ddbcd" />](https://github.com/user-attachments/assets/021f6671-2d7f-4635-91fb-88b2ab0ddbcd)

## 📌 Table of Contents

1. [🦜 Installation & Web UI](#installation)
2. [📦 Using the Python SDK](#sdk)
3. [🐳 High-Quality Server (Standard Mode)](#docker-remote)
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
   - **Option 1: CPU & macOS (minimal, torch-free) — recommended for maximum speed** — runs **v3 Turbo via ONNX**
     > 💡 *No GPU required. Installs only the lightweight ONNX stack; **v3 Turbo runs on CPU (48 kHz)** with default voices, voice cloning and emotion cues. PyTorch is never installed.*
     >
     > ⚡ **For the fastest CPU inference, install with `uv sync` — not `pip install`.** `uv sync` reproduces the locked environment that pins the optimized ONNX Runtime build, so you get maximum speed out of the box.
     >
     > 🍎 **macOS users: use this option too.** For v3 Turbo the torch-free ONNX path on the CPU is *faster* than the MPS/PyTorch build (`--group gpu`), so prefer `uv sync` for top speed on Apple Silicon.
     ```bash
     uv sync
     ```
   - **Option 2: GPU** — **v3 Turbo (PyTorch) + VieNeu-TTS v2 (GPU)**
     > 💡 *Requires a CUDA NVIDIA GPU (CUDA ≥ 12.8) or Apple Silicon MPS. [NVIDIA Toolkit](https://developer.nvidia.com/cuda-downloads) recommended. Adds the PyTorch stack so **v3 Turbo runs on GPU** and the **v1 / v2 (GPU)** models become available.*

     ```bash
     uv sync --group gpu
     ```

3. **Start the Web UI:**
   ```bash
   uv run vieneu-web
   ```
   Access the UI at `http://127.0.0.1:7860`.

---

## 📦 2. Using the Python SDK (vieneu) <a name="sdk"></a>

The `vieneu` SDK **defaults to VieNeu-TTS v3 Turbo (48 kHz)**. The minimal install is **torch-free**: on CPU everything runs on **ONNX Runtime** (PyTorch is never imported), and on a CUDA machine it auto-switches to the PyTorch engine. Older models (v1/v2) are available via the `[gpu]` extra.

### Quick Start
```bash
# Minimal, TORCH-FREE install — runs v3 Turbo on CPU via ONNX Runtime
pip install vieneu

```

```python
from vieneu import Vieneu
from time import time

# Default = v3 Turbo. CPU → ONNX (torch-free); GPU → PyTorch (auto-detected).
tts = Vieneu()

text = f"""[cười] Trời ơi, cái giọng nó tự nhiên mà nó mượt mà dã man, nghe không khác gì người thật luôn. Giờ thì tha hồ mà quẩy content với cả kho giọng nói đa dạng, đủ mọi sắc thái biểu cảm. Mọi người bật loa lên rồi cùng trải nghiệm thử với mình nhé!"""

start_time = time()
# 1. Default voice (Bình An) — 48 kHz, no reference needed
print("Bắt đầu sinh audio với giọng mặc định Bình An...")
audio = tts.infer(text)
tts.save(audio, "output.wav")
end_time = time()
print("Audio đã được sinh ra và được lưu vào file output.wav")
print(f"Thời gian sinh audio: {end_time - start_time:.2f} giây")
# 2. Built-in voices by name
print("Danh sách giọng nói có sẵn:")
for label, voice_id in tts.list_preset_voices():
    print(f"- {label} ({voice_id})")
print("Bắt đầu sinh audio với giọng Xuân Vĩnh...")
audio = tts.infer("Mình là Xuân Vĩnh nè!", voice="Xuân Vĩnh")
tts.save(audio, "output_Xuân Vĩnh.wav")
print("Audio đã được sinh ra và được lưu vào file output_Xuân Vĩnh.wav")
# # 3. Emotion / non-verbal cues — EXPERIMENTAL: [cười] [thở dài] [hắng giọng]
# audio = tts.infer("Nghe hay quá đi [cười]. Để mình nói tiếp [hắng giọng].", voice="Ngọc Linh")

# # 4. Instant voice cloning from a 3–5s reference clip
# audio = tts.infer("Đây là giọng được nhân bản tức thì.", ref_audio="my_voice.wav")
```

---

## 🐳 3. High-Quality Server (Standard Mode) <a name="docker-remote"></a>

Deploy VieNeu-TTS as a high-performance API Server (powered by LMDeploy) with a single command.

### 1. Run with Docker (Recommended)

**Requirement**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is required for GPU support.

**Start the Server with a Public Tunnel (No port forwarding needed):**
```bash
docker run --gpus all -p 23333:23333 -v huggingface_cache:/root/.cache/huggingface pnnbao/vieneu-tts:latest --tunnel
```

*   **Default**: The server loads the `VieNeu-TTS-v2` model for maximum quality.
*   **Tunneling**: The Docker image includes a built-in `bore` tunnel. Check the container logs to find your public address (e.g., `bore.pub:31631`).

### 2. Using the SDK (Remote Mode)

Once the server is running, you can connect from anywhere (Colab, Web Apps, etc.) without loading heavy models locally.

**Installation**:
```bash
pip install "vieneu[gpu]"
```

**Usage**:
```python
from vieneu import Vieneu
import os

# Configuration
REMOTE_API_BASE = 'http://your-server-ip:23333/v1'  # Or bore tunnel URL
REMOTE_MODEL_ID = "pnnbao-ump/VieNeu-TTS-v2"

# Initialization (LIGHTWEIGHT - only loads small codec locally)
# Default emotion is "natural" (conversational) - set emotion="storytelling" for storytelling mode
tts = Vieneu(mode='remote', api_base=REMOTE_API_BASE, model_name=REMOTE_MODEL_ID, emotion="natural")
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

| Model | Format | Device | Bilingual | Features | Speed |
|---|---|---|---|---|---|
| **VieNeu-TTS-v3-Turbo** *(early access)* | PyTorch | **GPU/CPU** | ✅ | **48 kHz, Default voices, Cloning, Emotion cues, Conversation** | **Fast (batched)** |
| **VieNeu-TTS-v2** | PyTorch | **GPU** | ✅ | **Podcast, En-Vi CS** | **Fast (LMDeploy)** |
| **VieNeu-v2-CPU** | GGUF/ONNX | **CPU/Edge** | ✅ | **Podcast, En-Vi CS** | **Extreme Speed** |
| **VieNeu-v2-Turbo** | GGUF/ONNX | **CPU/Edge** | ✅ | Lightweight En-Vi | **Ultra Fast** |
| **VieNeu-TTS (v1)** | PyTorch | GPU/CPU | ❌ | Stable (Vi only) | Standard |

> [!TIP]
> Use **Turbo v2** for AI assistants, chatbots, and real-time edge applications where speed is critical. Note: It may have stability issues with very short phrases (< 5 words).
> Use **GPU/Standard** (VieNeu-TTS v1/v2) for maximum audio quality and high-fidelity voice cloning.

---

## 🚀 5. Roadmap <a name="roadmap"></a>

- [x] **VieNeu-TTS-v2**: Full high-fidelity bilingual architecture with **Podcast Mode** and **Voice Cloning**.
- [x] **VieNeu-Codec**: Optimized neural codec for Vietnamese (ONNX).
- [x] **Turbo Voice Cloning**: Bringing instant cloning to the lightweight Turbo engine.
- [x] **VieNeu-TTS v3 Turbo (early access)**: New from-scratch 48 kHz architecture — built-in default voices (speaker tokens), experimental emotion cues, batched generation & multi-speaker conversation.
- [ ] **VieNeu-TTS v3 (full release)**: Complete v3 with finalized quality, stable emotion control, more default voices & streaming server.
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
  title        = {VieNeu-TTS-v2: Advanced Vietnamese Text-to-Speech with Podcast and Code-Switching Support},
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

This project uses [neucodec](https://huggingface.co/neuphonic/neucodec) (v1/v2) and [MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano) (v3 Turbo) for audio coding, and [sea-g2p](https://github.com/pnnbao97/sea-g2p) for text normalization and phonemization.

**Made with ❤️ for the Vietnamese TTS community**
