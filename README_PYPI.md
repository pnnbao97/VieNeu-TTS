# 🦜 VieNeu-TTS

**VieNeu-TTS** is an advanced on-device Vietnamese Text-to-Speech (TTS) with **instant voice cloning** and **English–Vietnamese bilingual** support. The SDK **defaults to VieNeu-TTS v3 Turbo (48 kHz)** and the minimal install is **torch-free** — on CPU it runs entirely on ONNX Runtime.

[![Hugging Face v3 Turbo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v3%20Turbo-red)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

## ✨ Key Features
- **v3 Turbo, 48 kHz** — high-fidelity, natural Vietnamese speech (default).
- **Torch-free on CPU** — minimal install runs on ONNX Runtime; PyTorch is never imported.
- **int8 backbone by default on CPU** — ~1.6× faster & ~4× smaller than fp32, quality preserved. Use `Vieneu(precision="fp32")` for max fidelity.
- **Built-in default voices** — call them by name, no reference clip needed.
- **Instant voice cloning** — clone any voice from 3–5s of audio.
- **Emotion cues** *(experimental)* — drop `[cười]`, `[thở dài]`, `[hắng giọng]` into the text.
- **Bilingual (En–Vi) code-switching**, fully offline.

---

## 📦 Install

**CPU (default)** — torch-free, runs v3 Turbo via ONNX Runtime. Most users want this:

```bash
pip install vieneu
```

**GPU (CUDA)** — only if you have an NVIDIA GPU. Install a CUDA build of PyTorch
**yourself first**. Batching then turns on automatically on CUDA — same API, no code change:

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=4.51"   # Qwen3 backbone + MOSS codec
pip install vieneu
```

> ℹ️ **When is GPU actually worth it?** The GPU win comes from **batching**, so it
> only pays off on **long text** (many chunks generated together in one forward —
> long-form or bulk synthesis). For **short text** the torch-free **CPU/ONNX** path
> is usually *faster* (there's no batch to fill). Use CPU for short, interactive
> calls; reach for GPU for long-form or high-throughput work.

---

## 🚀 Quick Start (Python SDK)

```python
from vieneu import Vieneu

# Default = v3 Turbo (48 kHz). GPU → PyTorch (auto-detected).
# On CPU the backbone runs int8 by default (fastest); pass precision="fp32" for max quality.
tts = Vieneu()                    # int8 backbone (default, fastest on CPU)
# tts = Vieneu(precision="fp32")  # fp32 backbone (max quality, slower on CPU)

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

# 4. ⚡ Batch on GPU: infer_batch() runs many texts in ONE batched forward — same API.
#    On a CUDA GPU the chunks from every text share each forward step (big throughput
#    win); on CPU it still works (no error), just sequentially. Batch caps at
#    max_batch_size (default 32; or infer_batch(..., batch_size=64); batch_size=1
#    disables). A single long infer() also auto-batches its own chunks. Uncomment to try:
#
# import time
# texts = [
#     "Chào cả nhà, hôm nay mình sẽ hướng dẫn các bạn cách cài đặt và sử dụng bộ giọng đọc mới.",
#     "Giọng nghe cực kỳ tự nhiên và truyền cảm, lại có thể chuyển đổi biểu cảm một cách linh hoạt.",
#     "Nếu thấy hữu ích, các bạn nhớ để lại một lượt thích và chia sẻ video này cho mọi người nhé!",
# ] * 10   # 30 texts — enough to fill the batch and really show the GPU throughput win
# t0 = time.time()
# audios = tts.infer_batch(texts, voice="Phạm Tuyên")
# elapsed = time.time() - t0
# total_audio = sum(len(a) for a in audios) / 48_000
# print(f"⚡ {len(texts)} texts | audio {total_audio:.1f}s | wall {elapsed:.1f}s | RTF {elapsed/total_audio:.3f}")
# for i, a in enumerate(audios):
#     tts.save(a, f"batch_{i}.wav")
```

### 🔊 Real-time streaming

v3 Turbo streams frame-by-frame (first audio ~300 ms, RTF < 1 on CPU). Streaming runs on the **ONNX/CPU** engine — the GPU/PyTorch engine is for **batch throughput**, not streaming, so pin `backend="onnx"` for realtime. Iterate `infer_stream`:

```python
tts = Vieneu(backend="onnx")   # force ONNX/CPU — the streaming path (int8)
for chunk in tts.infer_stream("Xin chào các bạn!", voice="Trúc Ly"):
    play(chunk)   # np.float32 @ 48 kHz, play/write as it arrives
```

A full FastAPI web demo is in [`apps/web_stream.py`](apps/web_stream.py) (`uv run python -m apps.web_stream` → http://localhost:8001).

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

---

## 🔬 Model Overview

| Model | Engine | Device | Sample Rate | Features |
|---|---|---|---|---|
| **VieNeu-TTS v3 Turbo** *(default)* | ONNX (CPU) / PyTorch (GPU) | CPU/GPU | 48 kHz | Default voices, cloning, emotion cues |

---

## 🤝 Support & Links
- **GitHub:** [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS)
- **Discord:** [Join our community](https://discord.gg/yJt8kzjzWZ)

**Made with ❤️ for the Vietnamese TTS community**
