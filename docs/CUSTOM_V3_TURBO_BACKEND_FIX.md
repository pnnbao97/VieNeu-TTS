# Thay đổi backend Custom Model v3 Turbo

## Vấn đề ban đầu

Khi chọn **Custom Model** và nhập đường dẫn:

```text
C:\Users\CaoPhuc\Projects\VieNeu-TTS-v3-Turbo
```

WebUI đưa model vào backend `standard.py`. Backend này gọi
`AutoModelForCausalLM.from_pretrained()`, nhưng model v3 Turbo sử dụng kiến trúc
riêng `vieneu_v3_turbo`, nên Transformers báo:

```text
ValueError: The checkpoint you are trying to load has model type
`vieneu_v3_turbo` but Transformers does not recognize this architecture.
```

## Các thay đổi

### 1. Nhận diện Custom v3 Turbo

File: `apps/gradio_main.py`

- Thêm `import json` ở dòng 17.
- Thêm biến `is_custom_v3_turbo`.
- Nhận diện model khi:
  - Tên đường dẫn/repository chứa `v3-turbo`; hoặc
  - `config.json` ở thư mục model hoặc `update\config.json` có:

    ```json
    {
      "model_type": "vieneu_v3_turbo"
    }
    ```

**Lý do:** WebUI cần biết đây là model v3 Turbo trước khi chọn backend.

### 2. Chuyển sang backend mới

File: `apps/gradio_main.py`

Custom v3 Turbo hiện được khởi tạo bằng:

```python
Vieneu(
    mode="v3turbo",
    backbone_repo=backbone_config["repo"],
    device=v3_device,
    precision=backbone_config.get("precision", "fp32"),
)
```

**Lý do:** `mode="v3turbo"` sử dụng engine riêng của VieNeu-TTS trong
`src/vieneu/v3turbo.py`, thay vì backend Transformers cũ.

### 3. Tắt LMDeploy cho Custom v3 Turbo

Custom v3 Turbo được ép sử dụng backend riêng và không đi qua LMDeploy.

**Lý do:** v3 Turbo có engine PyTorch/ONNX riêng; LMDeploy không phải backend
phù hợp cho checkpoint này.

### 4. Sửa lỗi phạm vi biến `json`

Đã xóa `import json` nằm bên trong `load_model()`.

**Lý do:** Python coi `json` là biến cục bộ của toàn bộ hàm nếu có import bên
trong hàm. Điều đó gây lỗi `UnboundLocalError` khi đọc `config.json` ở phía
trước. Chỉ giữ import `json` ở cấp module.

### 5. Cập nhật tài liệu

File: `docs/CUSTOM_MODEL_USAGE.md`

Đã bổ sung hướng dẫn nhập đường dẫn local của Custom v3 Turbo và mô tả cơ chế
nhận diện backend.

## Cách sử dụng sau khi sửa

Chạy WebUI:

```powershell
uv run vieneu-web
```

Trong giao diện:

1. Chọn **Custom Model**.
2. Nhập đường dẫn model v3 Turbo.
3. Bấm tải model.

Model local cần có checkpoint v3 Turbo hợp lệ, thông thường gồm thư mục
`update\config.json` và các file trọng số tương ứng.

## Kiểm tra

Đã chạy kiểm tra cú pháp:

```powershell
python -m py_compile apps\gradio_main.py
```

Kết quả: thành công.

## Commit liên quan

```text
26b40f0 Fix custom v3 Turbo backend routing
b0f692e Fix custom model config loading
```
