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

# Default = v3 Turbo (48 kHz). GPU → PyTorch (auto-detected).
tts = Vieneu()

# 1. Built-in voice by name — no reference needed
print("🔊 Generating speech...")
audio = tts.infer("Xin chào, đây là VieNeu-TTS.", voice="Trúc Ly")
tts.save(audio, "output.wav")
print("✅ Saved to output.wav")

# List the built-in voices
voices = tts.list_preset_voices()
print(f"\n🎙️  {len(voices)} built-in voices available:")
for label, voice_id in voices:
    print(f"  - {label} ({voice_id})")

# 2. Reading style: "tu_nhien" (natural) | "tin_tuc" (news) | "doc_truyen" (storytelling)
audio = tts.infer("Bản tin sáng nay.", voice="Phạm Tuyên", style="tin_tuc")

# 3. Emotion / non-verbal cues — EXPERIMENTAL: [cười] [thở dài] [hắng giọng]
audio = tts.infer("Nghe hay quá đi [cười].", voice="Trúc Ly")
```

### 🦜 Zero-shot Voice Cloning

Clone from a short clip; the reference is auto-denoised and trimmed to ≤ 8s.

```python
from vieneu import Vieneu
tts = Vieneu()

# Clone straight from a 3–8s clip
audio = tts.infer("Chào bạn, đây là giọng của tôi.", ref_audio="path/to/voice.wav", denoise=True)
tts.save(audio, "cloned.wav")

# Save a cloned voice and reuse it by name
tts.add_voice("Giọng của tôi", "path/to/voice.wav")
audio = tts.infer("Câu này dùng giọng đã lưu.", voice="Giọng của tôi")

# Just clean up a clip (no synthesis)
wav, sr = tts.denoise("noisy.wav", out_path="clean.wav")
```

> `denoise`, `add_voice`, and cloning require the PyTorch (GPU) engine; built-in voices work everywhere.

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
