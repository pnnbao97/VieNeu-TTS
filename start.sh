#!/usr/bin/env bash
# Khởi chạy VieNeu-TTS.
#
#   ./start.sh            # Web UI (Gradio) tại http://127.0.0.1:7860
#   ./start.sh web        # như trên
#   ./start.sh api        # API tương thích OpenAI (POST /v1/audio/speech) tại http://127.0.0.1:8000
#   ./start.sh stream     # Web Stream UI cũ (v1 CPU GGUF)
#   ./start.sh setup      # chỉ cài dependencies rồi thoát
#
# Tuỳ chọn (biến môi trường):
#   GPU=1                 cài thêm extra CUDA (uv sync --extra cuda) — chỉ cho máy NVIDIA
#   PORT=xxxx             đổi port (web: 7860, api: 8000)
#   HOST=0.0.0.0          mở ra mạng LAN (nên đặt VIENEU_WEB_AUTH / VIENEU_API_KEY)
#   VIENEU_PRECISION=int8 backbone int8 trên CPU (nhanh hơn, cần CPU có VNNI)
#
# Lưu ý: không nạp file .env — file đó dành cho Docker (HF_HOME=/root/..., đường dẫn Linux).

set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-web}"

log()  { printf '\033[0;36m>> %s\033[0m\n' "$*"; }
die()  { printf '\033[0;31m[ERROR] %s\033[0m\n' "$*" >&2; exit 1; }

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then return; fi
  [ -x "$HOME/.local/bin/uv" ] && export PATH="$HOME/.local/bin:$PATH" && return
  log "Chưa có uv — đang cài (https://astral.sh/uv)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "Cài uv thất bại. Cài thủ công rồi chạy lại."
}

sync_deps() {
  if [ "${GPU:-0}" = "1" ]; then
    log "Cài dependencies (GPU/CUDA)..."
    uv sync --extra cuda
  else
    log "Cài dependencies (CPU/macOS, torch-free)..."
    uv sync
  fi
}

ensure_uv

# Chỉ sync khi chưa có .venv hoặc lockfile mới hơn lần sync trước, để khởi động nhanh.
if [ "$MODE" = "setup" ] || [ ! -d .venv ] || [ uv.lock -nt .venv ] || [ pyproject.toml -nt .venv ]; then
  sync_deps
  touch .venv
fi

case "$MODE" in
  setup)
    log "Đã cài xong. Chạy: ./start.sh web | api"
    ;;
  web)
    export GRADIO_SERVER_NAME="${HOST:-127.0.0.1}"
    export GRADIO_SERVER_PORT="${PORT:-7860}"
    log "Web UI: http://${GRADIO_SERVER_NAME}:${GRADIO_SERVER_PORT} (lần đầu sẽ tải model từ Hugging Face)"
    exec uv run vieneu-web
    ;;
  api)
    export HOST="${HOST:-127.0.0.1}"
    export PORT="${PORT:-8000}"
    log "API: http://${HOST}:${PORT}/v1/audio/speech"
    exec uv run python -m apps.openai_speech
    ;;
  stream)
    log "Legacy Web Stream UI..."
    exec uv run vieneu-stream
    ;;
  -h|--help|help)
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *)
    die "Chế độ không hợp lệ: $MODE (dùng: web | api | stream | setup | help)"
    ;;
esac
