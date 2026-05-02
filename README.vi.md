# 🦜 VieNeu-TTS

[![Awesome](https://img.shields.io/badge/Awesome-NLP-green?logo=github)](https://github.com/keon/awesome-nlp)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yJt8kzjzWZ)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V1DjG-KdmurCAhvXrxxTLsa9tteDxSVO?usp=sharing)
[![Hugging Face v2 Turbo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v2%20Turbo-blue)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2-Turbo)
[![Hugging Face 0.3B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-0.3B-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B)

<img width="1087" height="710" alt="image" src="https://github.com/user-attachments/assets/5534b5db-f30b-4d27-8a35-80f1cf6e5d4d" />

**VieNeu-TTS** là mô hình chuyển đổi văn bản thành giọng nói (TTS) tiếng Việt tiên tiến chạy trên thiết bị, hỗ trợ **clone giọng nói tức thì** và khả năng đọc **song ngữ Anh-Việt**.

> [!IMPORTANT]
> **🚀 VieNeu-TTS-v2 Turbo:** Phiên bản mới nhất được tối ưu hóa cho CPU (GGUF) & GPU (LMDeploy), hỗ trợ đọc **song ngữ (Code-switching)** mượt mà và tốc độ xử lý cực nhanh.

## ✨ Tính năng nổi bật
- **Song ngữ Anh-Việt**: Chuyển đổi ngôn ngữ tự nhiên và mượt mà (Code-switching) ngay cả trong cùng một câu.
- **Clone giọng nói tức thì**: Clone bất kỳ giọng nói nào chỉ với **3-5 giây** âm thanh mẫu (**Turbo v2** & GPU mode).
- **Chế độ Turbo siêu nhanh**: Tối ưu hóa tối đa cho CPU/Mobile sử dụng GGUF và ONNX. Tốc độ cực nhanh!
- **Định danh AI**: Hệ thống đóng dấu bản quyền âm thanh ẩn để đảm bảo trách nhiệm nội dung (Audio Watermarking).
- **Sẵn sàng cho sản xuất**: Tạo âm thanh chất lượng cao 24 kHz, hoạt động hoàn toàn offline.

---

## 🦜 1. Cài đặt & Giao diện Web <a name="installation"></a>

### Thiết lập với `uv` (Khuyến nghị)
`uv` là cách nhanh nhất để quản lý các phụ thuộc.

1. **Clone Repo:**
   ```bash
   git clone https://github.com/pnnbao97/VieNeu-TTS.git
   cd VieNeu-TTS
   ```

2. **Cài đặt các phụ thuộc:**
   - **Lựa chọn 1: Tối giản (Turbo/CPU)** - Nhanh & Nhẹ
     ```bash
     uv sync
     ```
     > ⚠️ **Lưu ý:** Chế độ này chỉ hỗ trợ **VieNeu-TTS-v2-Turbo (CPU)** — chạy được trên mọi máy không cần GPU, nhưng **chất lượng âm thanh thấp hơn** so với Standard VieNeu-TTS (đặc biệt với câu ngắn < 5 từ). Phù hợp để thử nghiệm nhanh hoặc triển khai trên thiết bị yếu.
   - **Lựa chọn 2: Đầy đủ (GPU/Standard)** - Chất lượng cao & Cloning *(Dành cho người dùng GPU)*
     ```bash
     uv sync --group gpu
     ```
     > 💡 **Lưu ý:** Yêu cầu GPU NVIDIA hỗ trợ CUDA (hoặc Apple Silicon MPS). Kích hoạt backbone **Standard VieNeu-TTS** đầy đủ để đạt chất lượng âm thanh tối đa và clone giọng độ trung thực cao.

3. **Khởi chạy Giao diện Web:**
   ```bash
   uv run vieneu-web
   ```
   Truy cập giao diện tại `http://127.0.0.1:7860`. Model **Turbo v2** được chọn mặc định để bạn trải nghiệm ngay lập tức.

---

## 📦 2. Sử dụng Python SDK (vieneu) <a name="sdk"></a>

SDK `vieneu` hiện mặc định sử dụng **chế độ Standard** (0.3B GGUF + ONNX) khi dùng cục bộ, mang lại sự cân bằng hoàn hảo giữa chất lượng âm thanh cao và tốc độ xử lý thời gian thực trên cả CPU và GPU.

### Bắt đầu nhanh
```bash
# Cài đặt tối giản (Chỉ Turbo/CPU)
pip install vieneu

# Tùy chọn: llama-cpp-python đã build sẵn cho CPU (nếu cài bị lỗi)
pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/
```

### Ví dụ SDK (Python)
```python
from vieneu import Vieneu

# Khởi tạo chế độ Standard (Mặc định - Tối ưu 0.3B GGUF + ONNX)
# Chạy ngay lập tức trên CPU mà không cần cài đặt PyTorch!
tts = Vieneu()

# 1. Tổng hợp đơn giản (sử dụng giọng mặc định 'Xuân Vĩnh')
text = "Hệ thống điện chủ sử dụng alternating current because it is more efficient."
audio = tts.infer(text=text)
tts.save(audio, "output_Xuân Vĩnh.wav")

# 2. Sử dụng giọng nhân vật cụ thể
voices = tts.list_preset_voices()
for desc, voice_id in voices:
    print(f"Giọng: {desc} (ID: {voice_id})")

# Chọn giọng 'Phạm Tuyên (Nam miền Bắc)'
my_voice_id = voices[1][1] if len(voices) > 1 else voices[0][1]
voice_data = tts.get_preset_voice(my_voice_id)

audio_custom = tts.infer(text="Tôi đang nói bằng giọng của Bác sĩ Tuyên.", voice=voice_data)

# Lưu thành file
tts.save(audio_custom, "output_Phạm Tuyên.wav")
print("💾 Đã lưu file output_Phạm Tuyên.wav")
```

### 🚀 Chế độ Turbo (Song ngữ & Tốc độ cực nhanh)
Sử dụng `mode="turbo"` để đạt tốc độ xử lý nhanh nhất, đặc biệt tối ưu cho việc đọc song ngữ Anh-Việt (code-switching) trong thời gian thực.

```python
from vieneu import Vieneu

# Khởi tạo chế độ Turbo (v2-Turbo GGUF)
tts = Vieneu(mode="turbo")

# Turbo v2 hỗ trợ chuyển đổi Anh-Việt cực kỳ tự nhiên
text = "Hệ thống điện chủ sử dụng alternating current because it is more efficient."
audio = tts.infer(text=text)

tts.save(audio, "turbo_output.wav")
```

### 🦜 Clone giọng nói Zero-shot (SDK) <a name="cloning"></a>
Clone bất kỳ giọng nói nào chỉ với **3-5 giây** âm thanh mẫu.

> [!TIP]
> **Chế độ Turbo** được khuyến nghị cho việc clone giọng vì không yêu cầu văn bản mẫu (`ref_text`), trong khi **chế độ Standard** (mặc định) yêu cầu cung cấp `ref_text` để đạt độ chính xác cao hơn.

```python
from vieneu import Vieneu

# Sử dụng turbo mode để clone giọng dễ dàng (không cần ref_text)
tts = Vieneu(mode="turbo")

# 1. Trích xuất đặc trưng giọng nói (3-5 giây khuyến nghị)
my_voice = tts.encode_reference("examples/audio_ref/example.wav")

# 2. Tổng hợp với giọng đã clone
audio = tts.infer(
    text="Đây là giọng nói được clone trực tiếp bằng SDK của VieNeu-TTS.", 
    voice=my_voice
)

tts.save(audio, "cloned_voice.wav")
```

---

## 🐳 4. Docker & Remote Server <a name="docker-remote"></a>

Triển khai VieNeu-TTS dưới dạng API Server hiệu suất cao (được hỗ trợ bởi LMDeploy) chỉ bằng mã nguồn container duy nhất.

### 1. Chạy với Docker (Khuyến nghị)
```bash
docker run --gpus all -p 23333:23333 pnnbao/vieneu-tts:serve --tunnel
```

### 2. Sử dụng SDK (Chế độ Remote)
```python
from vieneu import Vieneu

# Khởi tạo (Cực kỳ NHẸ - chỉ tải codec nhỏ cục bộ)
tts = Vieneu(mode='remote', api_base='http://your-server-ip:23333/v1')

# Tổng hợp từ xa thông qua Server
audio = tts.infer(text="Chào bạn, đây là chế độ xử lý từ xa.")
tts.save(audio, "remote_output.wav")
```

---

## 🔬 5. Tổng quan mô hình <a name="backbones"></a>

| Model | Định dạng | Thiết bị | Song ngữ | Cloning | Tốc độ |
|---|---|---|---|---|---|
| **VieNeu-v2-Turbo** | GGUF/ONNX | **CPU/GPU** | ✅ | ✅ Có | **Cực nhanh** |
| **VieNeu-TTS-v2** | PyTorch | GPU | ✅ | ✅ Có | **Chuẩn** |
| **VieNeu-TTS 0.3B** | PyTorch | GPU/CPU | ❌ | ✅ Có | **Rất nhanh** |
| **VieNeu-TTS** | PyTorch | GPU/CPU | ❌ | ✅ Có | **Chuẩn** |

---

## 🚀 Lộ trình phát triển <a name="roadmap"></a>

- [x] **VieNeu-TTS-v2 Turbo**: Hỗ trợ chuyển đổi ngôn ngữ Anh-Việt (Bilingual).
- [x] **VieNeu-Codec**: Neural codec tối ưu cho tiếng Việt (định dạng ONNX).
- [x] **Turbo Voice Cloning**: Mang tính năng clone giọng nói lên engine Turbo siêu nhẹ.
- [ ] **VieNeu-TTS-v2 (Non-Turbo)**: Phiên bản đầy đủ chất lượng cao, tăng tốc qua **LMDeploy**.
- [ ] **Mobile SDK**: Hỗ trợ chính thức cho iOS và Android.

---

## 🤝 Hỗ trợ & Liên hệ <a name="support"></a>

- **Hugging Face:** [pnnbao-ump](https://huggingface.co/pnnbao-ump)
- **Discord:** [Tham gia cộng đồng](https://discord.gg/yJt8kzjzWZ)
- **Facebook:** [Pham Nguyen Ngoc Bao](https://www.facebook.com/pnnbao97)
- **Giấy phép:** Apache 2.0 (Sử dụng tự do).

---

**Được thực hiện với ❤️ dành cho cộng đồng TTS Việt Nam**
