# 🦜 VieNeu-TTS

**VieNeu-TTS** is an advanced on-device Vietnamese Text-to-Speech (TTS) with **instant voice cloning** and **English–Vietnamese bilingual** support. The SDK **defaults to VieNeu-TTS v3 Turbo (48 kHz)** and the minimal install is **torch-free** — on CPU it runs entirely on ONNX Runtime.

[![Hugging Face v3 Turbo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v3%20Turbo-red)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

## ✨ Key Features
- **v3 Turbo, 48 kHz** — high-fidelity, natural Vietnamese speech (default).
- **Torch-free on CPU** — minimal install runs on ONNX Runtime; PyTorch is never imported.
- **Built-in default voices** — call them by name, no reference clip needed.
- **Instant voice cloning** — clone any voice from 3–5s of audio.
- **Emotion cues** *(experimental)* — drop `[cười]`, `[thở dài]`, `[hắng giọng]` into the text.
- **Bilingual (En–Vi) code-switching**, fully offline.

---

## 📦 Install

```bash
# Minimal, TORCH-FREE — runs v3 Turbo on CPU via ONNX Runtime
pip install vieneu

# Optional: GPU + older backends (v1/v2 PyTorch & GGUF, v3 Turbo on GPU)
pip install "vieneu[gpu]"
```

---

## 🚀 Quick Start (Python SDK)

```python
from vieneu import Vieneu

# Default = v3 Turbo. CPU → ONNX (torch-free); GPU → PyTorch (auto-detected).
tts = Vieneu()

# 1. Default voice (Ngọc Lan) — 48 kHz, no reference needed
audio = tts.infer("Xin chào, đây là VieNeu-TTS phiên bản ba Turbo.")
tts.save(audio, "output.wav")

# 2. Built-in voices by name
for label, voice_id in tts.list_preset_voices():
    print(label, voice_id)
audio = tts.infer("Mình là Xuân Vĩnh nè!", voice="Xuân Vĩnh")

# 3. Emotion / non-verbal cues — EXPERIMENTAL: [cười] [thở dài] [hắng giọng]
audio = tts.infer("Nghe hay quá đi [cười]. Để mình nói tiếp [hắng giọng].", voice="Ngọc Linh")
```

### 🦜 Zero-shot Voice Cloning

```python
from vieneu import Vieneu
tts = Vieneu()

# Clone straight from a 3–5s clip — no reference transcript needed.
audio = tts.infer(text="Chào bạn, đây là giọng của tôi.", ref_audio="path/to/voice.wav")
tts.save(audio, "cloned.wav")
```

### Older models (v1 / v2 — requires `pip install "vieneu[gpu]"`)
```python
tts = Vieneu(mode="standard")   # v2 GGUF, bilingual, podcast
tts = Vieneu(mode="turbo")      # v2 Turbo, fastest
```

---

## 🔬 Model Overview

| Model | Engine | Device | Sample Rate | Features |
|---|---|---|---|---|
| **VieNeu-TTS v3 Turbo** *(default)* | ONNX (CPU) / PyTorch (GPU) | CPU/GPU | 48 kHz | Default voices, cloning, emotion cues |
| VieNeu-TTS v2 | PyTorch / GGUF | GPU/CPU | 24 kHz | Bilingual, podcast (`[gpu]`) |
| VieNeu-TTS v1 | PyTorch | GPU/CPU | 24 kHz | Stable, Vietnamese (`[gpu]`) |

---

## 🤝 Support & Links
- **GitHub:** [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS)
- **Discord:** [Join our community](https://discord.gg/yJt8kzjzWZ)

**Made with ❤️ for the Vietnamese TTS community**
