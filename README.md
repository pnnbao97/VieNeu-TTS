# 🦜 VieNeu-TTS

[![Awesome](https://img.shields.io/badge/Awesome-NLP-green?logo=github)](https://github.com/keon/awesome-nlp)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yJt8kzjzWZ)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b9PO-lcGZX9pEkEwQmu8MfhSnjxKrALW?usp=sharing)
[![Hugging Face v2 Turbo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v2%20Turbo-blue)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF)
[![Hugging Face VieNeu-TTS](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v1-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)

<img width="1087" height="710" alt="image" src="https://github.com/user-attachments/assets/5534b5db-f30b-4d27-8a35-80f1cf6e5d4d" />

**VieNeu-TTS** is an advanced on-device Vietnamese Text-to-Speech (TTS) model with **instant voice cloning** and **English-Vietnamese bilingual** support.

> [!IMPORTANT]
> **🚀 VieNeu-TTS-v2 Turbo:** The latest version is optimized for CPU & Low-end devices, featuring seamless **bilingual (Code-switching)** capabilities and ultra-fast inference.

## ✨ Key Features
- **Bilingual (English-Vietnamese)**: Smooth and natural transitions between languages powered by [sea-g2p](https://github.com/pnnbao97/sea-g2p).
- **Instant Voice Cloning**: Clone any voice with just **3-5 seconds** of reference audio (GPU/Standard mode).
- **Ultra-Fast Turbo Mode**: Optimized for CPU using GGUF and ONNX, requiring **NO GPU** and minimal RAM.
- **AI Identification**: Built-in audio watermarking for responsible AI content creation.
- **Production-Ready**: High-quality 24 kHz waveform generation, fully offline.

---

## 🦜 1. Installation & Web UI <a name="installation"></a>

### Setup with `uv` (Recommended)
`uv` is the fastest way to manage dependencies. [Install uv here](https://astral.sh/uv/install).

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
pip install vieneu
```

```python
from vieneu import Vieneu

# Initialize in Turbo mode (Default - Minimal dependencies)
tts = Vieneu(mode="turbo")

# Synthesize speech (uses default Southern Male voice 'Xuân Vĩnh')
text = "Trước đây, hệ thống điện chủ yếu sử dụng direct current, nhưng Tesla đã chứng minh rằng alternating current is more efficient."
audio = tts.infer(text=text)

# Save to file
tts.save(audio, "output.wav")
print("💾 Saved to output.wav")
```

### Advanced Modes
| Mode | Description | Requirements |
|---|---|---|
| `turbo` | (Default) Ultra-fast CPU inference | `onnxruntime`, `llama-cpp-python` |
| `remote` | Connect to a remote VieNeu API Server | `requests` |

---

## 🐳 3. Docker & Remote Server <a name="docker-remote"></a>

Deploy VieNeu-TTS as a high-performance API Server (GPU optimized).

```bash
# Run with GPU support
docker run --gpus all -p 23333:23333 pnnbao/vieneu-tts:serve --tunnel
```
Check container logs for your public `bore.pub` address.

---

## 🔬 4. Model Overview <a name="backbones"></a>

| Model | Format | Device | Bilingual | Cloning | Speed |
|---|---|---|---|---|---|
| **VieNeu-v2-Turbo** | GGUF/ONNX | **CPU**/GPU | ✅ | ❌ (Coming soon) | **Extreme** |
| **VieNeu-TTS-v2** | PyTorch | GPU | ✅ | ✅ Yes | **Standard** (Coming soon) |
| **VieNeu-TTS 0.3B** | PyTorch | GPU/CPU | ❌ | ✅ Yes | **Very Fast** |
| **VieNeu-TTS** | PyTorch | GPU/CPU | ❌ | ✅ Yes | **Standard** |

> [!TIP]
> Use **Turbo v2** for AI assistants, chatbots, and long-text reading on laptops. 
> Use **GPU/Standard** for high-quality voice cloning and artistic content.

---

## 🚀 Roadmap <a name="roadmap"></a>

- [x] **VieNeu-TTS-v2 Turbo**: English-Vietnamese code-switching support.
- [x] **VieNeu-Codec**: Optimized neural codec for Vietnamese (ONNX).
- [ ] **VieNeu-TTS-v2 (Non-Turbo)**: Full high-fidelity bilingual architecture with instant **Voice Cloning** and **LMDeploy** GPU acceleration support.
- [ ] **Turbo Voice Cloning**: Bringing instant cloning to the lightweight Turbo engine.
- [ ] **Mobile SDK**: Official support for Android/iOS deployment.

---

## 🤝 Support & Contact <a name="support"></a>

- **Hugging Face:** [pnnbao-ump](https://huggingface.co/pnnbao-ump)
- **Discord:** [Join our community](https://discord.gg/yJt8kzjzWZ)
- **Facebook:** [Pham Nguyen Ngoc Bao](https://www.facebook.com/pnnbao97)
- **License:** Apache 2.0 (Free to use).

---

**Made with ❤️ for the Vietnamese TTS community**
