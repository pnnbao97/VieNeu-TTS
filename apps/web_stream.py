"""
VieNeu-TTS v3 Turbo (int8) — CPU streaming demo (FastAPI).
==========================================================
Stream 48 kHz audio ngay khi generate, qua `V3TurboVieNeuTTS.infer_stream` (đường
ONNX/CPU int8 mặc định). Vì RTF < 1 (int8 nhanh hơn realtime), stream chạy mượt
không underrun — chỉ cần player prebuffer ~300–500ms.

    uv run python -m apps.web_stream        # http://localhost:8001

Public API dùng ở đây:
    tts = Vieneu()                                  # v3 Turbo int8, CPU
    for chunk in tts.infer_stream(text, voice="Trúc Ly"):
        ...                                         # np.float32 @ 48kHz, phát/ghi dần
"""
import time
import io
import wave
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel
import uvicorn

from vieneu import Vieneu

SAMPLE_RATE = 48_000
app = FastAPI()
tts = None


def load_model():
    global tts
    print("⏳ Loading VieNeu-TTS v3 Turbo (int8, CPU)...")
    # Mặc định: mode=v3turbo, precision=int8, device=auto→cpu. Truyền precision="fp32"
    # để đối chứng chất lượng (chậm hơn trên CPU).
    tts = Vieneu()  # == Vieneu(mode="v3turbo", precision="int8")
    print(f"✅ Ready. Backbone: int8 | intra_op threads: {getattr(tts.engine, 'ort_intra_op_threads', '?')}")


load_model()

# ── UI: HTML polished TỰ CHỨA (không CDN, dark-mode aware), khớp /stream + /voices.
HTML_CONTENT = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VieNeu v3 Turbo (int8) — CPU Streaming</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🐆</text></svg>">
<style>
:root{
  --bg:#f6f7fb; --card:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#e5e9f0;
  --accent:#10b981; --accent2:#0ea5e9; --accent-ink:#ffffff; --field:#f8fafc; --shadow:0 12px 40px rgba(15,23,42,.10);
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0b1220; --card:#111a2e; --ink:#e7edf7; --muted:#93a1b8; --line:#22304d;
  --accent:#10b981; --accent2:#38bdf8; --field:#0e1830; --shadow:0 12px 44px rgba(0,0,0,.45);
}}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;background:
  radial-gradient(1200px 600px at 15% -10%, color-mix(in srgb,var(--accent) 16%, transparent), transparent 60%),
  radial-gradient(1000px 500px at 110% 10%, color-mix(in srgb,var(--accent2) 14%, transparent), transparent 55%),
  var(--bg);
  color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  display:flex;align-items:flex-start;justify-content:center;padding:40px 16px}
.wrap{width:100%;max-width:640px}
.card{background:var(--card);border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow);padding:28px 26px}
.brand{display:flex;align-items:center;gap:12px;margin-bottom:4px}
.logo{font-size:30px;filter:drop-shadow(0 2px 6px rgba(16,185,129,.35))}
h1{font-size:20px;margin:0;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:13px;margin:2px 0 22px}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 22px}
.badge{font-size:12px;font-weight:600;padding:5px 10px;border-radius:999px;
  background:color-mix(in srgb,var(--accent) 14%, transparent);color:var(--accent);border:1px solid color-mix(in srgb,var(--accent) 30%, transparent)}
.badge.b2{background:color-mix(in srgb,var(--accent2) 14%, transparent);color:var(--accent2);border-color:color-mix(in srgb,var(--accent2) 30%, transparent)}
label{display:block;font-size:12.5px;font-weight:600;color:var(--muted);margin:0 0 7px;text-transform:uppercase;letter-spacing:.04em}
select,textarea{width:100%;background:var(--field);color:var(--ink);border:1px solid var(--line);
  border-radius:12px;padding:12px 13px;font:inherit;outline:none;transition:border-color .15s,box-shadow .15s}
select:focus,textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 22%, transparent)}
textarea{resize:none;overflow-y:auto;min-height:104px;max-height:340px}
.row{margin-bottom:18px}
.count{float:right;font-size:11px;color:var(--muted);font-weight:500;text-transform:none;letter-spacing:0}
.btn{width:100%;border:0;border-radius:13px;padding:14px;font:600 16px/1 system-ui;cursor:pointer;color:var(--accent-ink);
  background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 8px 22px color-mix(in srgb,var(--accent) 40%, transparent);
  display:flex;align-items:center;justify-content:center;gap:9px;transition:transform .08s,filter .15s}
.btn:hover{filter:brightness(1.06)} .btn:active{transform:translateY(1px)}
.btn:disabled{filter:grayscale(.4) brightness(.85);cursor:not-allowed;box-shadow:none}
.spin{width:16px;height:16px;border:2.5px solid rgba(255,255,255,.45);border-top-color:#fff;border-radius:50%;animation:sp .7s linear infinite;display:none}
@keyframes sp{to{transform:rotate(360deg)}}
.status{display:flex;align-items:center;gap:10px;margin-top:16px;min-height:22px;font-size:13.5px;color:var(--muted)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted);flex:none}
.dot.gen{background:#f59e0b;animation:pulse 1s infinite} .dot.play{background:var(--accent)} .dot.err{background:#ef4444}
@keyframes pulse{50%{opacity:.35}}
.ttfa{margin-left:auto;font-variant-numeric:tabular-nums;font-weight:700;color:var(--accent)}
audio{width:100%;margin-top:16px;border-radius:12px}
.foot{text-align:center;color:var(--muted);font-size:11.5px;margin-top:18px}
</style></head>
<body><div class="wrap"><div class="card">
  <div class="brand"><span class="logo">🐆</span><h1>VieNeu-TTS v3 Turbo</h1></div>
  <div class="sub">Text-to-Speech tiếng Việt · streaming thời gian thực trên CPU</div>
  <div class="badges">
    <span class="badge">⚡ backbone INT8</span>
    <span class="badge b2">🔊 48 kHz</span>
    <span class="badge">🖥️ CPU · không cần GPU</span>
  </div>
  <div class="row"><label>Giọng đọc</label><select id="voice"><option>Đang tải…</option></select></div>
  <div class="row"><label>Nội dung <span class="count" id="cnt">0</span></label>
    <textarea id="text" placeholder="Nhập văn bản tiếng Việt…">Xin chào các bạn, đây là bản stream thời gian thực chạy trên CPU bằng backbone int8. Nghe thử xem có mượt và tự nhiên không nhé!</textarea></div>
  <button class="btn" id="btn" onclick="go()"><span class="spin" id="spin"></span><span id="btxt">▶ Đọc (Stream)</span></button>
  <div class="status"><span class="dot" id="dot"></span><span id="st">Sẵn sàng</span><span class="ttfa" id="ttfa"></span></div>
  <audio id="au" controls></audio>
  <div class="foot">VieNeu-TTS · <code>pip install vieneu</code> · int8 mặc định trên CPU</div>
</div></div>
<script>
const $=id=>document.getElementById(id);
const txt=$('text'),cnt=$('cnt'),btn=$('btn'),spin=$('spin'),btxt=$('btxt'),dot=$('dot'),st=$('st'),ttfa=$('ttfa'),au=$('au');
const grow=()=>{txt.style.height='auto';txt.style.height=Math.min(txt.scrollHeight,340)+'px';};
txt.addEventListener('input',()=>{cnt.textContent=txt.value.length;grow();}); cnt.textContent=txt.value.length; grow();
fetch('/voices').then(r=>r.json()).then(v=>{const s=$('voice');s.innerHTML='';
  v.forEach(x=>{const o=document.createElement('option');o.value=x.id;o.textContent=x.name;s.add(o);});})
  .catch(e=>{setStatus('err','Lỗi tải giọng: '+e);});
function setStatus(kind,msg){dot.className='dot'+(kind?' '+kind:'');st.textContent=msg;}
function busy(on){btn.disabled=on;spin.style.display=on?'block':'none';btxt.textContent=on?'Đang tạo…':'▶ Đọc (Stream)';}
let t0=0,got=false;
function go(){
  const t=txt.value.trim(); if(!t)return;
  const v=$('voice').value;
  busy(true); got=false; ttfa.textContent=''; setStatus('gen','Đang tạo giọng…'); t0=performance.now();
  au.src='/stream?text='+encodeURIComponent(t)+'&voice_id='+encodeURIComponent(v);
  au.play().catch(()=>{});
}
au.addEventListener('playing',()=>{ if(!got){got=true; const ms=Math.round(performance.now()-t0);
  ttfa.textContent='⚡ '+ms+' ms'; } busy(false); setStatus('play','Đang phát'); });
au.addEventListener('ended',()=>setStatus('','Xong'));
au.addEventListener('error',()=>{busy(false);setStatus('err','Lỗi stream — xem log server');});
</script></body></html>"""


@app.get("/")
async def ui():
    return HTMLResponse(HTML_CONTENT)


@app.get("/favicon.ico")
async def favicon():
    # SVG emoji favicon → tab hiện 🐆, và hết log 404 favicon.
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y="82" font-size="82">🐆</text></svg>'
    return Response(svg, media_type="image/svg+xml")


@app.get("/voices")
async def voices():
    try:
        vs = tts.list_preset_voices()
        out = []
        for item in vs:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                label, vid = item
                out.append({"id": vid, "name": label})
            else:
                out.append({"id": str(item), "name": str(item)})
        return out or [{"id": "", "name": "(no preset voices)"}]
    except Exception as e:  # noqa: BLE001
        return [{"id": "", "name": f"⚠️ {e}"}]


def _pcm16(audio_f32: np.ndarray) -> bytes:
    return (np.asarray(audio_f32) * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


@app.get("/stream")
async def stream(text: str, voice_id: Optional[str] = None):
    """Stream 48 kHz WAV: header (nframes rất lớn để phát liên tục) rồi PCM16 theo chunk."""
    def gen():
        # WAV header 48 kHz mono; nframes huge → browser phát liền khi data tới.
        h = io.BytesIO()
        with wave.open(h, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            w.setnframes(1_000_000_000)
        yield h.getvalue()

        t0 = time.perf_counter()
        first_at = None
        n_chunks = 0
        emitted = 0
        for chunk in tts.infer_stream(text, voice=voice_id or None):
            if chunk is None or len(chunk) == 0:
                continue
            if first_at is None:
                first_at = time.perf_counter() - t0
                print(f"⚡ TTFA (time-to-first-audio): {first_at*1000:.0f} ms")
            n_chunks += 1
            emitted += len(chunk)
            yield _pcm16(chunk)
        if first_at is not None:
            gen_time = time.perf_counter() - t0
            audio_s = emitted / SAMPLE_RATE
            rtf = gen_time / audio_s if audio_s else 0
            print(f"✅ {n_chunks} chunks | audio {audio_s:.2f}s | gen {gen_time:.2f}s "
                  f"| RTF {rtf:.3f} ({1/rtf:.1f}x realtime)" if rtf else "")

    return StreamingResponse(gen(), media_type="audio/wav")


class StreamReq(BaseModel):
    text: str
    voice_id: Optional[str] = None


@app.post("/stream")
async def stream_post(req: StreamReq):
    return await stream(req.text, req.voice_id)


def main():
    print("🌍 Mở http://localhost:8001 để test VieNeu v3 Turbo (int8) streaming (CPU)")
    uvicorn.run(app, host="127.0.0.1", port=8001)


if __name__ == "__main__":
    main()
