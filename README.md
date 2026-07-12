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
> - **Built-in default voices** — stable and consistent, no reference clip needed.
> - **Reading styles**: natural, news, and storytelling.
> - **Emotion / non-verbal cues** *(experimental)*: drop `[cười]`, `[thở dài]`, `[hắng giọng]` straight into the text.
> - **Batched generation** (batch size up to 32), including a multi-speaker **Conversation** mode that batches the whole script regardless of speaker.
> - **Instant voice cloning** from a 3–8s clip, with automatic reference denoising.
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

> ⚡ **On CPU the backbone runs `int8` by default** — ~1.6× faster and ~4× smaller than fp32, with voice quality preserved. Want maximum fidelity instead? Pass `Vieneu(precision="fp32")` (slower on CPU). `precision` only affects the CPU/ONNX path; on GPU it's ignored (PyTorch).
>
> ```python
> tts = Vieneu()                    # int8 backbone (default, fastest on CPU)
> tts = Vieneu(precision="fp32")    # fp32 backbone (max quality, slower on CPU)
> ```

### Quick Start
```bash
pip install vieneu
```

```python
import time
from vieneu import Vieneu

# Default = v3 Turbo (48 kHz). GPU → PyTorch (auto-detected).
tts = Vieneu()

# 1. Built-in voice by name — no reference clip needed
print("🔊 Generating speech...")

start_time = time.time()
audio = tts.infer("[cười] Trời ơi, cái giọng nó tự nhiên mà nó mượt mà dã man, nghe không khác gì người thật luôn. Giờ thì tha hồ mà quẩy content với cả kho giọng nói đa dạng, đủ mọi sắc thái biểu cảm. Mọi người bật loa lên rồi cùng trải nghiệm thử với mình nhé!", voice="Phạm Tuyên")
elapsed_time = time.time() - start_time

tts.save(audio, "output.wav")
print("✅ Saved to output.wav")

# Tính RTF (Real-Time Factor)
sample_rate = 48000
audio_duration = len(audio) / sample_rate
rtf = elapsed_time / audio_duration

print(f"\n⏱️  Thời gian xử lý: {elapsed_time:.3f}s")
print(f"🎵 Thời lượng audio: {audio_duration:.3f}s")
print(f"📊 RTF: {rtf:.4f}  ({'nhanh hơn' if rtf < 1 else 'chậm hơn'} real-time {1/rtf:.2f}x)" if rtf > 0 else "")

# List the built-in voices
voices = tts.list_preset_voices()
print(f"\n🎙️  {len(voices)} built-in voices available:")
for label, voice_id in voices:
    print(f"  - {label} ({voice_id})")
```

#### Streaming (real-time) 🔊

v3 Turbo supports **frame-level streaming**: audio starts in ~300 ms and generation stays *ahead* of playback (RTF < 1 on CPU — ~2–3× on a laptop, ~7× on Apple Silicon), so it's ideal for realtime / interactive apps. Just iterate `infer_stream`:

```python
from vieneu import Vieneu
tts = Vieneu()                                    # int8 backbone, CPU
for chunk in tts.infer_stream("Xin chào các bạn!", voice="Minh Đức"):
    play(chunk)                                   # np.float32 @ 48 kHz — play/write as it arrives
```

A complete **FastAPI web streaming demo** (browser player, live time-to-first-audio, dark mode) is in [`apps/web_stream.py`](apps/web_stream.py):

```bash
uv run python -m apps.web_stream                  # → http://127.0.0.1:8001
```

#### Available Voices

The v3 Turbo engine includes **14 curated preset voices** covering **3 Vietnamese regions** (North, Central, South) with diverse genders and styles:

- **Northern (Bắc)**: Natural, news, storytelling styles
- **Central (Trung)**: Natural style (Quang Sơn, Ngọc Trân)
- **Southern (Nam)**: Natural, news, storytelling styles

Each voice supports **3 reading styles**: natural (`tu_nhien`), news (`tin_tuc`), and storytelling (`doc_truyen`).

### Reading style

Pick how the text is read with `style` (default `"tu_nhien"`):

| `style`        | Meaning       |
| -------------- | ------------- |
| `"tu_nhien"`   | Natural / conversational |
| `"tin_tuc"`    | News          |
| `"doc_truyen"` | Storytelling  |

```python
audio = tts.infer("Trận Caen là một trận đánh trong Chiến tranh Trăm Năm giữa Anh và Pháp diễn ra vào ngày 26 tháng 7 năm 1346 khi quân viễn chinh Anh dưới sự chỉ huy của Edward III tấn công thành Caen do quân Pháp nắm giữ.", voice="Phạm Tuyên", style="tin_tuc")
```

### Emotion cues (experimental)

Inline tags are supported anywhere in the text: `[cười]` (chuckle), `[thở dài]` (sigh), `[hắng giọng]` (clear throat).

```python
audio = tts.infer("Nghe hay quá đi [cười]. Để mình nói tiếp [hắng giọng].", voice="Trúc Ly")
```

### Voice cloning

Clone any voice from a short reference clip. The clip is cleaned up automatically
(background noise removed, and trimmed to ≤ 8 seconds) before cloning — keep
`denoise=True` unless your clip is already clean.

```python
audio = tts.infer(
    "Đây là giọng được nhân bản tức thì.",
    ref_audio="my_voice.wav",   # a 3–8s reference clip
    denoise=True,               # default; set False if the clip is already clean
    style="doc_truyen",
)
tts.save(audio, "cloned.wav")
```

### Save & reuse a cloned voice

Register a reference once with `add_voice`, then use it by name like a built-in voice.

```python
# Enroll a voice (denoises + extracts the speaker profile once)
tts.add_voice("Giọng của tôi", "my_voice.wav")

# Now reuse it anywhere, including the conversation mode
audio = tts.infer("Câu này dùng giọng đã lưu.", voice="Giọng của tôi")

# Persist your voices so they load next session
tts.save_voices()                 # writes to the default voices file
# tts.remove_voice("Giọng của tôi")

# Add a voice you already cleaned yourself → skip denoising
tts.add_voice("Giọng sạch", "already_clean.wav", denoise=False)
```

### Clean up a clip on its own

Get the denoised audio without synthesizing anything (e.g. to inspect or store it):

```python
wav, sr = tts.denoise("noisy.wav", out_path="clean.wav")   # 44.1 kHz mono
```

> **Note:** `denoise`, `add_voice`, and voice cloning currently require the PyTorch
> (GPU) engine. Built-in voices work everywhere.

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
| **VieNeu-TTS-v3-Turbo** *(early access)* | PyTorch/ONNX | **GPU/CPU** | ✅ | **48 kHz, Default voices, Cloning, Emotion cues, Conversation** | **Fast (batched)** |
| **VieNeu-TTS-v2** | PyTorch | **GPU** | ✅ | **Podcast, En-Vi CS** | **Fast (LMDeploy)** |
| **VieNeu-v2-CPU** | GGUF/ONNX | **CPU/Edge** | ✅ | **Podcast, En-Vi CS** | **Extreme Speed** |
| **VieNeu-v2-Turbo** | GGUF/ONNX | **CPU/Edge** | ✅ | Lightweight En-Vi | **Ultra Fast** |
| **VieNeu-TTS (v1)** | PyTorch | GPU/CPU | ❌ | Stable (Vi only) | Standard |

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

<a href="https://www.star-history.com/?repos=pnnbao97%2FVieNeu-TTS&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=pnnbao97/VieNeu-TTS&type=date&theme=dark&legend=top-left&sealed_token=_tsrXAlf4KfIavj2geHlH_hTI3Vt6YrjXHEaMwuo0iSbNTryaItVoXiSBw4rkyhZkoiSWWW0hopsrddCfpPZciHd5LO53sX8MaFFCcmE6fQC75nhf9wzR61_wpp7cFSxvW9jvlsUOfK4XLSVhPY_ZhKMdqJXboryhvjYulUuawE5jGkAUjQRSFKKl8GK" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=pnnbao97/VieNeu-TTS&type=date&legend=top-left&sealed_token=_tsrXAlf4KfIavj2geHlH_hTI3Vt6YrjXHEaMwuo0iSbNTryaItVoXiSBw4rkyhZkoiSWWW0hopsrddCfpPZciHd5LO53sX8MaFFCcmE6fQC75nhf9wzR61_wpp7cFSxvW9jvlsUOfK4XLSVhPY_ZhKMdqJXboryhvjYulUuawE5jGkAUjQRSFKKl8GK" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=pnnbao97/VieNeu-TTS&type=date&legend=top-left&sealed_token=_tsrXAlf4KfIavj2geHlH_hTI3Vt6YrjXHEaMwuo0iSbNTryaItVoXiSBw4rkyhZkoiSWWW0hopsrddCfpPZciHd5LO53sX8MaFFCcmE6fQC75nhf9wzR61_wpp7cFSxvW9jvlsUOfK4XLSVhPY_ZhKMdqJXboryhvjYulUuawE5jGkAUjQRSFKKl8GK" />
 </picture>
</a>

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
