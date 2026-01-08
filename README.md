# VieNeu-TTS

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/pnnbao97/VieNeu-TTS)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.5B-yellow)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.3B-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.3B--GGUF-green)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B-q8-gguf)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/mQWr4cp3)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V1DjG-KdmurCAhvXrxxTLsa9tteDxSVO?usp=sharing) 

<img width="899" height="615" alt="Untitled" src="https://github.com/user-attachments/assets/7eb9b816-6ab7-4049-866f-f85e36cb9c6f" />

**VieNeu-TTS** is an advanced on-device Vietnamese Text-to-Speech (TTS) model with **instant voice cloning**.

This project features two core architectures trained on the [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h) dataset:
- **VieNeu-TTS (0.5B):** An enhanced model fine-tuned from the NeuTTS Air architecture for maximum stability.
- **VieNeu-TTS-0.3B:** A specialized model **trained from scratch**, delivering 2x faster inference and ultra-low latency.

These represent a significant upgrade from the previous VieNeu-TTS-140h with the following improvements:

- **Enhanced pronunciation**: More accurate and stable Vietnamese pronunciation
- **Code-switching support**: Seamless transitions between Vietnamese and English
- **Better voice cloning**: Higher fidelity and speaker consistency
- **Real-time synthesis**: 24 kHz waveform generation on CPU or GPU
- **Multiple model formats**: Support for PyTorch, GGUF Q4/Q8 (CPU optimized), and ONNX codec

VieNeu-TTS delivers production-ready speech synthesis fully offline.

**Author:** Phạm Nguyễn Ngọc Bảo

---

[<img width="600" height="595" alt="VieNeu-TTS" src="https://github.com/user-attachments/assets/6b32df9d-7e2e-474f-94c8-43d6fa586d15" />](https://github.com/user-attachments/assets/6b32df9d-7e2e-474f-94c8-43d6fa586d15)

---

## 🔬 Model Overview

- **Backbone:** 
  - **VieNeu-TTS (0.5B):** Qwen-0.5B fine-tuned from [NeuTTS Air](https://huggingface.co/neuphonic/neutts-air).
  - **VieNeu-TTS-0.3B:** Custom 0.3B model **trained from scratch**, optimized for extreme speed (2x faster).
- **Audio codec:** NeuCodec (torch implementation; ONNX & quantized variants supported)
- **Context window:** 2,048 tokens shared by prompt text and speech tokens
- **Output watermark:** Enabled by default
- **Training data:** [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h) — 443,641 curated Vietnamese samples (Used for both versions).

### Model Variants

| Model                   | Format  | Device  | Quality    | Speed                   |
| ----------------------- | ------- | ------- | ---------- | ----------------------- |
| VieNeu-TTS              | PyTorch | GPU/CPU | ⭐⭐⭐⭐⭐ | Very Fast with lmdeploy |
| VieNeu-TTS-0.3B         | PyTorch | GPU/CPU | ⭐⭐⭐⭐   | **Ultra Fast (2x)**     |
| VieNeu-TTS-q8-gguf      | GGUF Q8 | CPU/GPU | ⭐⭐⭐⭐   | Fast                    |
| VieNeu-TTS-q4-gguf      | GGUF Q4 | CPU/GPU | ⭐⭐⭐     | Very Fast               |
| VieNeu-TTS-0.3B-q8-gguf | GGUF Q8 | CPU/GPU | ⭐⭐⭐⭐   | **Ultra Fast (1.5x)**   |
| VieNeu-TTS-0.3B-q4-gguf | GGUF Q4 | CPU/GPU | ⭐⭐⭐     | **Extreme Speed (2x)**  |

**Recommendations:**

- **GPU users**: Use `VieNeu-TTS` (PyTorch) for best quality
- **CPU users**: Use `VieNeu-TTS-0.3B-q4-gguf` for fastest inference or `VieNeu-TTS-0.3B-q8-gguf` for best CPU quality.
- **Streaming**: Only GGUF models support streaming inference (Requires `llama-cpp-python >= 0.3.16`)

---

## ✅ Todo & Status

- [x] Publish safetensor artifacts
- [x] Release GGUF Q4 / Q8 models
- [x] Release datasets (1000h and 140h)
- [x] Enable streaming on GPU
- [x] Provide Dockerized setup
- [ ] Release fine-tuning code

---

## 🏁 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/pnnbao97/VieNeu-TTS.git
cd VieNeu-TTS
```

### 2. Install eSpeak NG (Required)
Phonemizer requires eSpeak NG to function.

- **Windows:** Download installer from [eSpeak NG Releases](https://github.com/espeak-ng/espeak-ng/releases) (Recommended: `.msi`).
- **macOS:** `brew install espeak`
- **Ubuntu/Debian:** `sudo apt install espeak-ng`
- **Arch Linux:** `paru -S aur/espeak-ng`

---

### 3. Environment Setup (Choose ONE method)

#### Method 1: Standard with `uv` (Recommended)
This is the fastest and most reliable way to manage dependencies.

**A. Install `uv`** (If you haven't already):
- **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **Linux/macOS:** `curl -LsSf https://astral.sh/uv/install.sh | sh`

**B. Choose your hardware:**

**Option A: For GPU Users (NVIDIA 30xx/40xx/50xx)**

> [!IMPORTANT]
> **Update your NVIDIA Drivers & Install CUDA Toolkit!**
> This project uses **CUDA 12.8**. Please ensure your NVIDIA driver is up-to-date (support CUDA 12.8 or newer) to avoid compatibility issues, especially on RTX 30 series.
>
> To use `lmdeploy`, you **MUST** install the **NVIDIA GPU Computing Toolkit**: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

```bash
uv sync
```

**Option B: For CPU-only Users**

1. Switch to CPU configuration:
   ```bash
   # Windows:
   ren pyproject.toml pyproject.toml.bak
   copy pyproject.toml.cpu pyproject.toml
   
   # Linux/macOS:
   mv pyproject.toml pyproject.toml.bak
   cp pyproject.toml.cpu pyproject.toml
   ```
2. Install dependencies:
   ```bash
   uv sync
   ```

**C. Run the Application:**
```bash
uv run gradio_app.py
```

Then access the Web UI at `http://127.0.0.1:7860`.

---

#### Method 2: Automatic with Makefile (Alternative)
Best if you have `make` installed (standard on Linux/macOS, or via Git Bash on Windows). It handles configuration swaps automatically.

- **Setup GPU:** `make setup-gpu`
- **Setup CPU:** `make setup-cpu`
- **Run Demo:** `make demo`


Then access the Web UI at `http://127.0.0.1:7860`.

**Optional dependencies:**

- **For GGUF models (GPU):**
  *Requires `llama-cpp-python >= 0.3.16`*
  ```bash
  $env:CMAKE_ARGS="-DGGML_CUDA=on"
  uv pip install "llama-cpp-python>=0.3.16" --force-reinstall --no-cache-dir
  ```

---

## 🐋 Docker Deployment

For a quick start or production deployment without manually installing dependencies, use Docker.

### Quick Start

Copy .env.example to .env

```
cp .env.example .env
```

Build and start container

```bash
# Run with CPU
docker compose --profile cpu up

# Run with GPU (requires NVIDIA Container Toolkit)
docker compose --profile gpu up
```

Access the Web UI at `http://localhost:7860`.

For detailed deployment instructions, including production setup, see [docs/Deploy.md](docs/Deploy.md).

---

## 📦 Project Structure

```
VieNeu-TTS/
├── examples/
│   ├── infer_long_text.py     # CLI for long-form synthesis (chunked)
│   └── sample_long_text.txt   # Example paragraph for testing
├── gradio_app.py              # Local Gradio web demo with LMDeploy support
├── main.py                    # Basic batch inference script
├── config.yaml                # Configuration for models, codecs, and voices
├── output_audio/              # Generated audio (created when running scripts)
├── sample/                    # Reference voices (audio + transcript + codes)
│   ├── Bình (nam miền Bắc).wav/txt/pt
│   ├── Đoan (nữ miền Nam).wav/txt/pt
│   ├── Dung (nữ miền Nam).wav/txt/pt
│   ├── Hương (nữ miền Bắc).wav/txt/pt
│   ├── Ly (nữ miền Bắc).wav/txt/pt
│   ├── Ngọc (nữ miền Bắc).wav/txt/pt
│   ├── Nguyên (nam miền Nam).wav/txt/pt
│   ├── Sơn (nam miền Nam).wav/txt/pt
│   ├── Tuyên (nam miền Bắc).wav/txt/pt
│   └── Vĩnh (nam miền Nam).wav/txt/pt
├── utils/
│   ├── __init__.py
│   ├── core_utils.py          # Text chunking utilities
│   ├── normalize_text.py      # Vietnamese text normalization pipeline
│   ├── phonemize_text.py      # Text to phoneme conversion
│   └── phoneme_dict.json      # Phoneme dictionary
├── vieneu_tts/
│   ├── __init__.py            # Exports VieNeuTTS and FastVieNeuTTS
│   └── vieneu_tts.py          # Core VieNeuTTS implementation (VieNeuTTS & FastVieNeuTTS)
├── README.md
├── requirements.txt           # Basic dependencies (legacy)
├── pyproject.toml             # Project configuration with full dependencies (UV)
└── uv.lock                    # UV lock file for dependency management
```

---

## 📚 References

- [GitHub Repository](https://github.com/pnnbao97/VieNeu-TTS)
- [Hugging Face Model (0.5B)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
- [Hugging Face Model (0.3B)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B)
- [VieNeuTTS Fine-tuning Guide](https://github.com/pnnbao-ump/VieNeuTTS/blob/main/finetune.ipynb)
- [VieNeuCodec dataset](https://huggingface.co/datasets/pnnbao-ump/VieNeuCodec-dataset)

---

## 📄 License

- **VieNeu-TTS (0.5B):** Original terms (Apache 2.0).
- **VieNeu-TTS-0.3B:** Released under **CC BY-NC 4.0** (Non-Commercial). 
  - This version is currently **experimental**.
  - **Commercial use is prohibited** without authorization. Please contact the author for commercial licensing.

---

## 📑 Citation

```bibtex
@misc{vieneutts2026,
  title        = {VieNeu-TTS: Vietnamese Text-to-Speech with Instant Voice Cloning},
  author       = {Pham Nguyen Ngoc Bao},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/pnnbao-ump/VieNeu-TTS}}
}
```

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "Add amazing feature"`
4. Push the branch: `git push origin feature/amazing-feature`
5. Open a pull request

---

## 📞 Support

- GitHub Issues: [github.com/pnnbao97/VieNeu-TTS/issues](https://github.com/pnnbao97/VieNeu-TTS/issues)
- Hugging Face: [huggingface.co/pnnbao-ump](https://huggingface.co/pnnbao-ump)
- Discord: [Join with us](https://discord.gg/mQWr4cp3)
- Facebook: [Phạm Nguyễn Ngọc Bảo](https://www.facebook.com/bao.phamnguyenngoc.5)

---

## 🙏 Acknowledgements

This project builds upon [NeuTTS Air](https://huggingface.co/neuphonic/neutts-air) for the original 0.5B model. The 0.3B version is a custom architecture trained from scratch using the [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h) dataset.

---

**Made with ❤️ for the Vietnamese TTS community**
