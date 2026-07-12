# 🦜 VieNeu-TTS

[![Awesome](https://img.shields.io/badge/Awesome-NLP-green?logo=github)](https://github.com/keon/awesome-nlp)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yJt8kzjzWZ)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b9PO-lcGZX9pEkEwQmu8MfhSnjxKrALW?usp=sharing)
[![Hugging Face VieNeu-TTS-v2](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v2-blue)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2)
[![Hugging Face VieNeu-TTS](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v1-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)

<img width="1087" height="710" alt="image" src="https://github.com/user-attachments/assets/5534b5db-f30b-4d27-8a35-80f1cf6e5d4d" />

**VieNeu-TTS-v2** là thế hệ tiếp theo của mô hình chuyển đổi văn bản thành giọng nói (TTS) tiếng Việt chạy trên thiết bị, hỗ trợ **10.000+ giờ dữ liệu** huấn luyện song ngữ, **clone giọng nói tức thì**, và chế độ **Podcast/Hội thoại** chuyên dụng.

> [!IMPORTANT]
> **🚀 VieNeu-TTS-v2 đã ra mắt!**
> Kiến trúc song ngữ chất lượng cao (high-fidelity) hiện đã sẵn sàng với:
> - **10.000+ Giờ dữ liệu:** Độ tự nhiên vượt trội trong cả tiếng Anh và tiếng Việt.
> - **Chế độ Podcast & Đối thoại:** Hỗ trợ đa người nói với các sắc thái biểu cảm.
> - **Zero-shot Cloning:** Clone bất kỳ giọng nói nào chỉ trong 3-5 giây trên tất cả các biến thể v2.

## ✨ Tính năng nổi bật
- **Huấn luyện 10.000+ giờ**: Được huấn luyện trên tập dữ liệu Anh-Việt khổng lồ cho ngữ điệu giống hệt con người.
- **Song ngữ (En-Vi) Code-switching**: Chuyển đổi ngôn ngữ mượt mà ngay trong câu.
- **Chế độ Podcast & Hội thoại**: Hỗ trợ đối thoại đa người nói với khả năng tự động nhận diện nhân vật.
- **Clone giọng nói tức thì**: Clone bất kỳ giọng nói nào chỉ với **3-5 giây** âm thanh mẫu.
- **Hiệu suất cực nhanh**: Được tối ưu hóa cho **GPU (LMDeploy)** và **CPU (GGUF/ONNX)**.
- **Sẵn sàng cho sản xuất**: Tạo âm thanh chất lượng cao 24 kHz, hoạt động hoàn toàn offline.

[<img width="600" height="595" alt="VieNeu-TTS Demo" src="https://github.com/user-attachments/assets/021f6671-2d7f-4635-91fb-88b2ab0ddbcd" />](https://github.com/user-attachments/assets/021f6671-2d7f-4635-91fb-88b2ab0ddbcd)

## 📌 Mục lục

1. [🦜 Cài đặt & Giao diện Web](#installation)
2. [📦 Sử dụng Python SDK](#sdk)
3. [🐳 Server Chất lượng cao (Standard Mode)](#docker-remote)
4. [🔬 Tổng quan mô hình](#backbones)
5. [🚀 Lộ trình phát triển](#roadmap)
6. [🤝 Hỗ trợ & Liên hệ](#support)
7. [📑 Trích dẫn](#citation)

---

## 🦜 1. Cài đặt & Giao diện Web <a name="installation"></a>

### Thiết lập với `uv` (Khuyến nghị)
`uv` là cách nhanh nhất để quản lý các phụ thuộc.
```bash
# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. **Clone Repo:**
   ```bash
   git clone https://github.com/pnnbao97/VieNeu-TTS.git
   cd VieNeu-TTS
   ```

2. **Cài đặt các phụ thuộc:**
   - **Lựa chọn 1: CPU & macOS (tối giản, không cần torch) — khuyến nghị để đạt tốc độ tối đa** — chạy **v3 Turbo bằng ONNX**
     > 💡 *Không cần GPU. Chỉ cài bộ ONNX nhẹ; **v3 Turbo chạy trên CPU (48 kHz)** với giọng mặc định, voice cloning và tag cảm xúc. Hoàn toàn không cài PyTorch.*
     >
     > ⚡ **Để CPU chạy nhanh nhất, hãy cài bằng `uv sync` — đừng dùng `pip install`.** `uv sync` dựng lại đúng môi trường đã khóa (lockfile) với bản ONNX Runtime đã tối ưu, nhờ đó đạt tốc độ tối đa ngay từ đầu.
     >
     > 🍎 **Người dùng macOS: cũng dùng lựa chọn này.** Với v3 Turbo, đường ONNX không-torch chạy trên CPU *nhanh hơn* bản MPS/PyTorch (`--group gpu`), nên hãy ưu tiên `uv sync` để đạt tốc độ cao nhất trên Apple Silicon.
     ```bash
     uv sync
     ```
   - **Lựa chọn 2: GPU** — **v3 Turbo chạy trên GPU (PyTorch)**
     > 💡 *Yêu cầu GPU NVIDIA CUDA (CUDA ≥ 12.8) hoặc Apple Silicon MPS. Khuyến nghị cài [NVIDIA Toolkit](https://developer.nvidia.com/cuda-downloads). Thêm bộ PyTorch để **v3 Turbo chạy trên GPU** — trên CUDA suy luận được **batch tự động** (cùng API, không đổi code).*

     ```bash
     uv sync --group gpu
     ```

3. **Khởi chạy Giao diện Web:**
   ```bash
   uv run vieneu-web
   ```
   Truy cập giao diện tại `http://127.0.0.1:7860`.

---

## 📦 2. Sử dụng Python SDK (vieneu) <a name="sdk"></a>

SDK `vieneu` **mặc định dùng VieNeu-TTS v3 Turbo (48 kHz)**. Bản cài tối giản **không cần torch**: trên CPU mọi thứ chạy bằng **ONNX Runtime** (PyTorch không bao giờ được import), còn trên máy CUDA nó tự chuyển sang engine PyTorch — nơi suy luận được **batch tự động** (cùng API, không đổi code).

> ⚡ **Trên CPU, backbone chạy `int8` theo mặc định** — nhanh ~1.6× và nhẹ ~4× so với fp32, chất giọng vẫn giữ nguyên. Cần chất lượng tối đa? Truyền `Vieneu(precision="fp32")` (chậm hơn trên CPU). `precision` chỉ ảnh hưởng đường CPU/ONNX; trên GPU nó bị bỏ qua (PyTorch).
>
> ```python
> tts = Vieneu()                    # backbone int8 (mặc định, nhanh nhất trên CPU)
> tts = Vieneu(precision="fp32")    # backbone fp32 (chất lượng tối đa, chậm hơn trên CPU)
> ```

### Bắt đầu nhanh
**CPU (mặc định)** — không cần torch, chạy v3 Turbo bằng ONNX Runtime. Đa số người dùng chọn cái này:

```bash
pip install vieneu
```

**GPU (CUDA)** — chỉ khi bạn có GPU NVIDIA. Tự cài bản PyTorch CUDA **trước** (không có extra `[gpu]`). Trên CUDA, batch tự bật — cùng API, không đổi code:

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=4.51"   # Qwen3 backbone + MOSS codec
pip install vieneu
```

> ℹ️ **Khi nào GPU thật sự đáng dùng?** Lợi thế của GPU đến từ **batch**, nên chỉ
> đáng khi **text dài** (nhiều chunk chạy chung một forward — đọc dài, tổng hợp hàng
> loạt). Với **text ngắn**, đường **CPU/ONNX** không-torch thường *nhanh hơn* (không
> có gì để lấp batch). Dùng CPU cho câu ngắn, tương tác; dùng GPU cho đọc dài hoặc
> khối lượng lớn.

```python
from vieneu import Vieneu

# Mặc định = v3 Turbo (48 kHz). GPU → PyTorch (tự nhận diện).
tts = Vieneu()

# 1. Giọng dựng sẵn theo tên — không cần audio mẫu
print("🔊 Đang sinh giọng nói...")
audio = tts.infer("Xin chào, đây là VieNeu-TTS.", voice="Trúc Ly")
tts.save(audio, "output.wav")
print("✅ Đã lưu vào output.wav")

# Liệt kê các giọng dựng sẵn
voices = tts.list_preset_voices()
print(f"\n🎙️  Có {len(voices)} giọng dựng sẵn:")
for label, voice_id in voices:
    print(f"  - {label} ({voice_id})")

# 2. ⚡ Batch trên GPU: infer_batch() chạy nhiều text trong MỘT lần forward — cùng API.
#    Trên GPU CUDA, các chunk của mọi text dùng chung mỗi bước forward (throughput cao
#    hơn nhiều); trên CPU vẫn CHẠY ĐƯỢC (không lỗi), chỉ là tuần tự. Batch tối đa
#    max_batch_size (mặc định 32; hoặc infer_batch(..., batch_size=64); batch_size=1 để
#    tắt). Một infer() cho text dài cũng tự batch các chunk. Bỏ comment để thử (nên dùng GPU):
#
# import time
# texts = [
#     "Chào cả nhà, hôm nay mình sẽ hướng dẫn các bạn cách cài đặt và sử dụng bộ giọng đọc mới.",
#     "Giọng nghe cực kỳ tự nhiên và truyền cảm, lại có thể chuyển đổi biểu cảm một cách linh hoạt.",
#     "Nếu thấy hữu ích, các bạn nhớ để lại một lượt thích và chia sẻ video này cho mọi người nhé!",
# ] * 10   # 30 câu — đủ lấp đầy batch để thấy rõ sức mạnh throughput của GPU
# t0 = time.time()
# audios = tts.infer_batch(texts, voice="Phạm Tuyên")
# elapsed = time.time() - t0
# total_audio = sum(len(a) for a in audios) / 48_000
# print(f"⚡ {len(texts)} câu | audio {total_audio:.1f}s | thời gian {elapsed:.1f}s | RTF {elapsed/total_audio:.3f}")
# for i, a in enumerate(audios):
#     tts.save(a, f"batch_{i}.wav")
```

### Streaming thời gian thực 🔊

v3 Turbo hỗ trợ **streaming theo frame**: audio ra sau ~300 ms và generator luôn *chạy vượt* player (RTF < 1 trên CPU — ~2–3× trên laptop, ~7× trên Apple Silicon), rất hợp cho ứng dụng realtime / tương tác. Streaming chạy trên engine **ONNX/CPU** — độ trễ audio đầu thấp, theo từng frame; engine GPU/PyTorch sinh ra để **batch throughput**, không dành cho streaming, nên hãy ép `backend="onnx"` cho realtime. Chỉ cần lặp `infer_stream`:

```python
from vieneu import Vieneu
tts = Vieneu(backend="onnx")                      # ép ONNX/CPU — đường dành cho streaming (int8)
for chunk in tts.infer_stream("Xin chào các bạn!", voice="Trúc Ly"):
    play(chunk)                                   # np.float32 @ 48 kHz — phát/ghi ngay khi có
```

Bản demo **web streaming FastAPI** đầy đủ (player trên trình duyệt, hiện time-to-first-audio, dark mode) nằm ở [`apps/web_stream.py`](apps/web_stream.py):

```bash
uv run python -m apps.web_stream                  # → http://localhost:8001
```

> Engine chia chunk thích ứng (chunk đầu ~320 ms cho độ trễ thấp, rồi phình tới ~2 s khi đã dư lead). Vì RTF < 1 nên lead chỉ tăng dần → player prebuffer ~300 ms là dư, không underrun.

### Phong cách đọc

Chọn cách đọc bằng `style` (mặc định `"tu_nhien"`):

| `style`        | Ý nghĩa        |
| -------------- | -------------- |
| `"tu_nhien"`   | Tự nhiên / hội thoại |
| `"tin_tuc"`    | Tin tức        |
| `"doc_truyen"` | Kể chuyện      |

```python
audio = tts.infer("Bản tin sáng nay.", voice="Phạm Tuyên", style="tin_tuc")
```

### Tag cảm xúc (thử nghiệm)

Chèn trực tiếp trong văn bản: `[cười]`, `[thở dài]`, `[hắng giọng]`.

```python
audio = tts.infer("Nghe hay quá đi [cười]. Để mình nói tiếp [hắng giọng].", voice="Trúc Ly")
```

> [!TIP]
> Temperature ~0.8 ổn định nhất.

### 🦜 Clone giọng nói Zero-shot (SDK) <a name="cloning"></a>
Clone bất kỳ giọng nào từ một clip ngắn. Clip mẫu được **tự khử nhiễu nền** và **cắt còn ≤ 8 giây** trước khi clone — cứ để `denoise=True` trừ khi clip đã sạch.

```python
from vieneu import Vieneu

tts = Vieneu()

# Clone trực tiếp từ clip mẫu (3–8 giây)
audio = tts.infer(
    text="Đây là giọng được nhân bản tức thì.",
    ref_audio="examples/audio_ref/example.wav",
    denoise=True,          # mặc định; đặt False nếu clip đã sạch
    style="doc_truyen",
)
tts.save(audio, "cloned_voice.wav")
```

#### Lưu & tái dùng giọng đã clone
Đăng ký clip một lần bằng `add_voice`, sau đó gọi theo tên như giọng dựng sẵn (dùng được cả ở chế độ Hội thoại).

```python
# Đăng ký giọng (tự denoise + trích hồ sơ giọng một lần)
tts.add_voice("Giọng của tôi", "my_voice.wav")

audio = tts.infer("Câu này dùng giọng đã lưu.", voice="Giọng của tôi")

# Lưu lại để lần sau vẫn còn
tts.save_voices()
# tts.remove_voice("Giọng của tôi")

# Thêm giọng bạn đã tự làm sạch → bỏ qua bước denoise
tts.add_voice("Giọng sạch", "already_clean.wav", denoise=False)
```

#### Chỉ khử nhiễu một clip
Lấy audio đã khử nhiễu mà không tổng hợp gì (để nghe/lưu lại):

```python
wav, sr = tts.denoise("noisy.wav", out_path="clean.wav")   # 44.1 kHz mono
```

> **Lưu ý:** `denoise`, `add_voice` và voice cloning hiện cần engine PyTorch (GPU). Giọng dựng sẵn chạy ở mọi nơi.

---

## 🐳 3. Server Chất lượng cao (Standard Mode) <a name="docker-remote"></a>

Triển khai VieNeu-TTS dưới dạng API Server hiệu suất cao (được hỗ trợ bởi LMDeploy) chỉ bằng một câu lệnh duy nhất.

### 1. Chạy với Docker (Khuyến nghị)

**Yêu cầu**: Cần cài đặt [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) để hỗ trợ GPU.

**Khởi chạy Server với Đường hầm công khai (Không cần mở cổng modem):**
```bash
docker run --gpus all -p 23333:23333 -v huggingface_cache:/root/.cache/huggingface pnnbao/vieneu-tts:latest --tunnel
```

*   **Mặc định**: Server sẽ tải model `VieNeu-TTS-v2` để đạt chất lượng tối đa.
*   **Tunneling**: Docker image tích hợp sẵn đường hầm `bore`. Kiểm tra container logs để tìm địa chỉ công khai của bạn (VD: `bore.pub:31631`).

### 2. Sử dụng SDK (Chế độ Remote)

Khi server đã chạy, bạn có thể kết nối từ bất kỳ đâu (Colab, Web App, v.v.) mà không cần tải các model nặng cục bộ.

**Cài đặt**:
```bash
pip install "vieneu[legacy]"
```

**Sử dụng**:
```python
from vieneu import Vieneu
import os

# Cấu hình
REMOTE_API_BASE = 'http://your-server-ip:23333/v1'  # Hoặc URL từ bore tunnel
REMOTE_MODEL_ID = "pnnbao-ump/VieNeu-TTS-v2"

# Khởi tạo (Cực kỳ NHẸ - chỉ tải codec nhỏ cục bộ)
# Cảm xúc mặc định là "natural" (tự nhiên) - đặt emotion="storytelling" cho chế độ kể chuyện
tts = Vieneu(mode='remote', api_base=REMOTE_API_BASE, model_name=REMOTE_MODEL_ID, emotion="natural")
os.makedirs("outputs", exist_ok=True)

# Liệt kê các giọng mẫu trên server
available_voices = tts.list_preset_voices()
for desc, name in available_voices:
    print(f"   - {desc} (ID: {name})")

# Sử dụng giọng cụ thể (chọn động giọng thứ hai)
if available_voices:
    _, my_voice_id = available_voices[1]
    voice_data = tts.get_preset_voice(my_voice_id)
    audio_spec = tts.infer(text="Chào bạn, tôi đang nói bằng giọng của bác sĩ Tuyên.", voice=voice_data)
    tts.save(audio_spec, f"outputs/remote_{my_voice_id}.wav")
    print(f"💾 Đã lưu kết quả tại: outputs/remote_{my_voice_id}.wav")

# Tổng hợp chuẩn (dùng giọng mặc định)
text_input = "Chế độ remote giúp tích hợp VieNeu vào ứng dụng Web hoặc App cực nhanh mà không cần GPU tại máy khách."
audio = tts.infer(text=text_input)
tts.save(audio, "outputs/remote_output.wav")
print("💾 Đã lưu kết quả remote_output.wav")

# Clone giọng Zero-shot (Mã hóa âm thanh cục bộ, gửi code lên server)
if os.path.exists("examples/audio_ref/example_ngoc_huyen.wav"):
    cloned_audio = tts.infer(
        text="Đây là giọng nói được clone và xử lý thông qua VieNeu Server.",
        ref_audio="examples/audio_ref/example_ngoc_huyen.wav",
        ref_text="Tác phẩm dự thi bảo đảm tính khoa học, tính đảng, tính chiến đấu, tính định hướng."
    )
    tts.save(cloned_audio, "outputs/remote_cloned_output.wav")
    print("💾 Đã lưu kết quả remote_cloned_output.wav")
```
*Chi tiết xem tại: [examples/main_remote.py](examples/main_remote.py)*

### Quy chuẩn Voice Preset (v1.0)
VieNeu-TTS sử dụng quy chuẩn chính thức `vieneu.voice.presets` để định nghĩa các tài nguyên giọng nói có thể tái sử dụng. Chỉ các tệp `voices.json` tuân theo quy chuẩn này mới đảm bảo tương thích với VieNeu-TTS SDK ≥ v1.x.

### 3. Cấu hình Nâng cao

Tùy chỉnh server để chạy các phiên bản cụ thể hoặc các model đã được fine-tune của riêng bạn.

**Chạy model 0.3B (Nhanh hơn):**
```bash
docker run --gpus all pnnbao/vieneu-tts:serve --model pnnbao-ump/VieNeu-TTS-0.3B --tunnel
```

**Serve model đã Fine-tuned cục bộ:**
Nếu bạn đã merge LoRA adapter, hãy mount thư mục đầu ra của bạn vào container:
```bash
# Linux / macOS
docker run --gpus all \
  -v $(pwd)/finetune/output:/workspace/models \
  pnnbao/vieneu-tts:serve \
  --model /workspace/models/merged_model --tunnel
```

---

## 🔬 4. Tổng quan mô hình <a name="backbones"></a>

| Model | Engine | Thiết bị | Sample Rate | Tính năng |
|---|---|---|---|---|
| **VieNeu-TTS v3 Turbo** *(mặc định)* | ONNX (CPU) / PyTorch (GPU) | CPU/GPU | 48 kHz | Giọng dựng sẵn, clone giọng, cảm xúc |

> [!TIP]
> Trên **CPU**, backbone chạy `int8` mặc định (nhanh nhất); dùng `Vieneu(precision="fp32")` nếu cần chất lượng tối đa. Trên **GPU (CUDA)**, suy luận **tự động batch** — cùng API, không đổi code.

---

## 🚀 5. Lộ trình phát triển <a name="roadmap"></a>

- [x] **VieNeu-TTS-v2**: Kiến trúc song ngữ chất lượng cao đầy đủ với **Chế độ Podcast** và **Clone giọng nói**.
- [x] **VieNeu-Codec**: Neural codec tối ưu cho tiếng Việt (ONNX).
- [x] **Turbo Voice Cloning**: Mang tính năng clone giọng nói tức thì lên engine Turbo siêu nhẹ.
- [ ] **Mobile SDK**: Hỗ trợ chính thức cho việc triển khai trên Android/iOS.

---

## 🤝 6. Hỗ trợ & Liên hệ <a name="support"></a>

- **Hugging Face:** [pnnbao-ump](https://huggingface.co/pnnbao-ump)
- **Discord:** [Tham gia cộng đồng](https://discord.gg/yJt8kzjzWZ)
- **Facebook:** [Phạm Nguyễn Ngọc Bảo](https://www.facebook.com/pnnbao97)
- **Giấy phép:** Apache 2.0 (Sử dụng tự do).

---
## 📑 7. Trích dẫn <a name="citation"></a>

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

## 🤝 Người đóng góp

Cảm ơn tất cả những người tuyệt vời đã đóng góp cho dự án này!

<a href="https://github.com/pnnbao97/VieNeu-TTS/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pnnbao97/VieNeu-TTS" />
</a>

---

## 🙏 Lời cảm ơn

Dự án này sử dụng [neucodec](https://huggingface.co/neuphonic/neucodec) để giải mã âm thanh và [sea-g2p](https://github.com/pnnbao97/sea-g2p) để chuẩn hóa văn bản và phiên âm.

**Được thực hiện với ❤️ dành cho cộng đồng TTS Việt Nam**
