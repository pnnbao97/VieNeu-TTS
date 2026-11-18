# 🎯 Hướng dẫn sử dụng Custom Model (LoRA, GGUF, Finetune)

## 📖 Giới thiệu

VieNeu-TTS hỗ trợ tải và sử dụng các mô hình tùy chỉnh từ HuggingFace hoặc đường dẫn cục bộ (Local Path), bao gồm:

1.  **LoRA Adapters**: Model fine-tune bằng phương pháp LoRA. (Hỗ trợ tăng tốc với **LMDeploy**!)
2.  **Custom Finetune Models**: Model fine-tune đầy đủ (Full Finetune) dựa trên VieNeu-TTS.
3.  **GGUF Quantized Models**: Model lượng tử hóa GGUF (chạy trên CPU/llama.cpp).

### ✅ Custom v3 Turbo

Bạn cũng có thể nhập đường dẫn local của checkpoint v3 Turbo vào **Custom Model**,
ví dụ `C:\Users\CaoPhuc\Projects\VieNeu-TTS-v3-Turbo`. WebUI sẽ nhận diện
`model_type: vieneu_v3_turbo` (hoặc tên có `v3-Turbo`) và tự dùng backend v3 Turbo
riêng, không đi qua Transformers/LMDeploy cũ.

---

## 📌 Mục lục

- [🚀 Cách sử dụng LoRA Adapter (với LMDeploy)](#-cách-sử-dụng-lora-adapter-với-lmdeploy)
- [📦 Cách sử dụng GGUF Model (CPU)](#-cách-sử-dụng-gguf-model-cpu)
- [🔧 Khắc phục sự cố](#-khắc-phục-sự-cố)
- [💡 Tips & Lưu ý](#-tips--lưu-ý)

---

## 🚀 Cách sử dụng LoRA Adapter (với LMDeploy)

Đây là cách tối ưu nhất để chạy giọng fine-tune với tốc độ cao.

### Bước 1: Cấu hình Model
1. Tại mục **🦜 Backbone**, chọn **`Custom Model`**.
2. Một bảng nhập liệu sẽ hiện ra bên dưới.
3. **Custom Model ID**: Nhập Repo ID trên HuggingFace (hoặc đường dẫn folder local).
   - Ví dụ: `pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen`
4. **HF Token** (Tùy chọn): Nhập HuggingFace Access Token nếu repo là **Private**.
5. **Base Model**: Chọn Base Model tương ứng mà LoRA đã được train trên đó.
   - *Mẹo*: Hệ thống thường tự động phát hiện (ví dụ tên có "0.3" sẽ chọn bản 0.3B).
6. **🚀 Optimize with LMDeploy**: **NÊN TICK** chọn để kích hoạt tăng tốc.

### Bước 2: Tải Model
1. Click nút **🔄 Tải Model**.
2. Hệ thống sẽ tự động:
   - Tải LoRA Adapter và Base Model.
   - **Merge** (gộp) chúng lại với nhau (sử dụng GPU nếu có để tăng tốc).
   - Lưu model đã merge vào bộ nhớ đệm (`merged_models_cache/`).
   - Load model đã optimization bằng **LMDeploy**.
3. *Lưu ý*: Quá trình Merge chỉ chạy **một lần đầu tiên**. Lần sau sẽ load cực nhanh từ cache.

### Bước 3: Sử dụng Custom Voice
1. Sau khi load xong, chuyển sang tab **"🦜 Custom Voice"**.
2. **Audio Reference**: Upload file audio mẫu (tốt nhất là file nằm trong tập train của LoRA).
3. **Text Reference**: Nhập chính xác nội dung văn bản của file audio đó.
   - *Lưu ý*: Text phải khớp từng dấu câu, từng chữ.
4. Nhập văn bản cần đọc vào ô chính và nhấn **🎵 Bắt đầu**.

---

## 📦 Cách sử dụng GGUF Model (CPU)

Dành cho máy không có GPU NVIDIA hoặc muốn chạy nhẹ nhàng trên CPU.

1. Tại mục **Backbone**, chọn **`Custom Model`**.
2. **Custom Model ID**: Nhập Repo ID chứa "gguf" trong tên hoặc file.
   - Ví dụ: `pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf` (hoặc repo custom của bạn).
3. Hệ thống sẽ tự động nhận diện từ khóa "gguf" và chuyển sang backend **llama-cpp** (Standard).
4. Các bước tiếp theo tương tự như trên.

---

## 🔧 Khắc phục sự cố

### 1. Lỗi "LMDeploy Init Error" hoặc model không load được
- **Nguyên nhân**: Có thể do thiếu file tokenizer hoặc config trong quá trình merge.
- **Cách khắc phục**:
  - Xóa folder cache của model đó trong thư mục `merged_models_cache/`.
  - Thử tải lại để hệ thống merge lại từ đầu.

### 2. Tiếng nói bị rè hoặc không giống giọng mẫu
- Đảm bảo **Audio Reference** và **Text Reference** khớp nhau 100%.
- Kiểm tra xem bạn đã chọn đúng **Base Model** chưa? (LoRA train trên 0.3B không thể chạy trên base 0.5B và ngược lại).

### 3. Model Private không tải được
- Hãy chắc chắn bạn đã nhập đúng **HF Token** có quyền `read` vào ô HF Token.

---

## 💡 Tips & Lưu ý

1.  **Cache**: Các model LoRA sau khi merge sẽ chiếm dụng dung lượng ổ cứng trong `merged_models_cache`. Bạn có thể xóa thủ công các folder trong đó nếu muốn giải phóng bộ nhớ.
2.  **Tự động hóa**: Nếu bạn nhập Repo ID `pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen`, hệ thống sẽ tự động điền các thông tin và file mẫu cho bạn (Demo mode).
3.  **Tốc độ**: Với LMDeploy, tốc độ sinh giọng sẽ nhanh gấp nhiều lần so với backend cũ. Hãy tận dụng GPU!
