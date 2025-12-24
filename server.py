"""
VieNeu-TTS Combined Server

Runs both Gradio UI and REST API on the same server.
- Gradio UI: http://localhost:7860
- REST API: http://localhost:7860/api/v1/...
- API Docs: http://localhost:7860/api/v1/docs

Run with: uv run server.py
"""

import gradio as gr
print("⏳ Đang khởi động VieNeu-TTS Server...")

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import soundfile as sf
import tempfile
import torch
import os
import time
import yaml
import uuid
import numpy as np

from vieneu_tts import VieNeuTTS
from utils.core_utils import split_text_into_chunks

# --- CONFIG ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f) or {}

BACKBONE_CONFIGS = _config.get("backbone_configs", {})
CODEC_CONFIGS = _config.get("codec_configs", {})
VOICE_SAMPLES = _config.get("voice_samples", {})
_text_settings = _config.get("text_settings", {})
MAX_CHARS_PER_CHUNK = _text_settings.get("max_chars_per_chunk", 256)

# --- SHARED STATE ---
# This will be set by Gradio app when model is loaded
shared_tts_engine = None
shared_config = {"backbone": None, "codec": None, "device": None}


def set_shared_engine(engine, config):
    """Called by Gradio to share the loaded model with API."""
    global shared_tts_engine, shared_config
    shared_tts_engine = engine
    shared_config = config


def get_shared_engine():
    """Get the shared TTS engine."""
    return shared_tts_engine, shared_config


def auto_load_suitable_model():
    """Automatically load a suitable model based on available hardware."""
    global shared_tts_engine, shared_config
    
    # Check if model is already loaded
    if shared_tts_engine is not None:
        print("✅ Model already loaded")
        return True
    
    print("🔍 No model loaded. Auto-loading suitable model...")
    
    try:
        # Use load_model_with_params with None to auto-select
        result = load_model_with_params(
            backbone_choice=None,
            codec_choice=None,
            device_choice="Auto"
        )
        return result["success"]
    except (ValueError, Exception) as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error auto-loading model: {e}")
        return False


def load_model_with_params(backbone_choice: Optional[str] = None, 
                           codec_choice: Optional[str] = None,
                           device_choice: str = "Auto"):
    """Load model with specific parameters or auto-select if not provided."""
    global shared_tts_engine, shared_config
    
    # Determine suitable model based on hardware if not provided
    has_gpu = torch.cuda.is_available()
    
    # Select backbone
    if backbone_choice is None or backbone_choice not in BACKBONE_CONFIGS:
        if has_gpu:
            if "VieNeu-TTS-q8-gguf" in BACKBONE_CONFIGS:
                backbone_choice = "VieNeu-TTS-q8-gguf"
            elif "VieNeu-TTS (GPU)" in BACKBONE_CONFIGS:
                backbone_choice = "VieNeu-TTS (GPU)"
            else:
                backbone_choice = list(BACKBONE_CONFIGS.keys())[0]
        else:
            # CPU: Prefer PyTorch model for better compatibility (especially M1 Macs)
            # GGUF models can have issues generating speech tokens on some systems
            if "VieNeu-TTS (GPU)" in BACKBONE_CONFIGS:
                backbone_choice = "VieNeu-TTS (GPU)"  # Works on CPU too, despite the name
            elif "VieNeu-TTS-q8-gguf" in BACKBONE_CONFIGS:
                backbone_choice = "VieNeu-TTS-q8-gguf"
            elif "VieNeu-TTS-q4-gguf" in BACKBONE_CONFIGS:
                backbone_choice = "VieNeu-TTS-q4-gguf"
            else:
                backbone_choice = list(BACKBONE_CONFIGS.keys())[0]
    
    # Select codec
    if codec_choice is None or codec_choice not in CODEC_CONFIGS:
        if has_gpu:
            codec_choice = "NeuCodec (Standard)"
        else:
            # CPU: Use standard codec (works better with PyTorch backbone)
            # ONNX codec requires pre-encoded codes which may not always be available
            codec_choice = "NeuCodec (Standard)"
    
    # Validate choices
    if backbone_choice not in BACKBONE_CONFIGS:
        raise ValueError(f"Invalid backbone: {backbone_choice}. Available: {list(BACKBONE_CONFIGS.keys())}")
    if codec_choice not in CODEC_CONFIGS:
        raise ValueError(f"Invalid codec: {codec_choice}. Available: {list(CODEC_CONFIGS.keys())}")
    
    print(f"Loading model:")
    print(f"   Backbone: {backbone_choice}")
    print(f"   Codec: {codec_choice}")
    print(f"   Device: {device_choice}")
    
    try:
        # Cleanup existing model if any
        if shared_tts_engine is not None:
            print("Unloading existing model...")
            del shared_tts_engine
            shared_tts_engine = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Get configs
        backbone_config = BACKBONE_CONFIGS[backbone_choice]
        codec_config = CODEC_CONFIGS[codec_choice]
        
        # Determine devices based on device_choice
        if device_choice == "Auto":
            use_gpu = has_gpu
        elif device_choice == "CUDA":
            use_gpu = has_gpu
            if not has_gpu:
                raise ValueError("CUDA requested but GPU not available")
        else:  # CPU
            use_gpu = False
        
        # Determine device strings
        if use_gpu:
            if "gguf" in backbone_choice.lower():
                backbone_device = "gpu"
            else:
                backbone_device = "cuda"
            
            if "ONNX" in codec_choice:
                codec_device = "cpu"
            else:
                codec_device = "cuda"
        else:
            if "gguf" in backbone_choice.lower():
                backbone_device = "cpu"
            else:
                backbone_device = "cpu"
            
            if "ONNX" in codec_choice:
                codec_device = "cpu"
            else:
                codec_device = "cpu"
        
        # Directly instantiate the model
        from vieneu_tts import VieNeuTTS
        
        print(f"   Loading backbone on {backbone_device}, codec on {codec_device}...")
        tts = VieNeuTTS(
            backbone_repo=backbone_config["repo"],
            backbone_device=backbone_device,
            codec_repo=codec_config["repo"],
            codec_device=codec_device
        )
        
        # Set shared engine
        device = "cuda" if use_gpu else "cpu"
        set_shared_engine(tts, {
            "backbone": backbone_choice,
            "codec": codec_choice,
            "device": device
        })
        
        print(f"Model loaded successfully!")
        print(f"   Backbone: {backbone_choice}")
        print(f"   Codec: {codec_choice}")
        print(f"   Device: {device}")
        
        return {
            "success": True,
            "message": "Model loaded successfully",
            "backbone": backbone_choice,
            "codec": codec_choice,
            "device": device
        }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        print(f"Error loading model: {error_msg}")
        # Don't raise HTTPException here - let the caller decide
        raise ValueError(f"Failed to load model: {error_msg}")


def ensure_model_loaded():
    """Ensure model is loaded; do NOT auto-load."""
    engine, _ = get_shared_engine()
    if engine is None:
        print("Model not loaded. Please load a model manually.")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Load a model via /api/v1/load-model or the Gradio UI first."
        )
    return get_shared_engine()


# --- API MODELS ---
class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize (Vietnamese)")
    voice: str = Field(
        default="Vĩnh (nam miền Nam)",
        description="Preset voice name"
    )


class ModelStatus(BaseModel):
    loaded: bool
    backbone: Optional[str] = None
    codec: Optional[str] = None
    device: Optional[str] = None
    available_voices: List[str]


class LoadModelRequest(BaseModel):
    backbone: Optional[str] = Field(
        default=None,
        description="Backbone model name. If not provided, will auto-select based on hardware."
    )
    codec: Optional[str] = Field(
        default=None,
        description="Codec model name. If not provided, will auto-select based on hardware."
    )
    device: Optional[str] = Field(
        default="Auto",
        description="Device: 'Auto', 'CPU', or 'CUDA'"
    )


class LoadModelResponse(BaseModel):
    success: bool
    message: str
    backbone: Optional[str] = None
    codec: Optional[str] = None
    device: Optional[str] = None


# --- API SETUP ---
api_app = FastAPI(
    title="VieNeu-TTS API",
    description="Vietnamese Text-to-Speech REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_reference_data(voice_name: str):
    """Load reference audio codes and text for a voice."""
    if voice_name not in VOICE_SAMPLES:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    
    engine, _ = ensure_model_loaded()
    
    voice_info = VOICE_SAMPLES[voice_name]
    text_path = voice_info["text"]
    codes_path = voice_info.get("codes")
    audio_path = voice_info["audio"]
    
    with open(text_path, "r", encoding="utf-8") as f:
        ref_text = f.read()
    
    if codes_path and os.path.exists(codes_path):
        ref_codes = torch.load(codes_path, map_location="cpu", weights_only=True)
    else:
        ref_codes = engine.encode_reference(audio_path)
    
    if isinstance(ref_codes, torch.Tensor):
        ref_codes = ref_codes.cpu().numpy()
    
    return ref_codes, ref_text


@api_app.get("/")
async def api_root():
    return {"status": "ok", "service": "VieNeu-TTS API", "docs": "/api/v1/docs"}


@api_app.get("/status", response_model=ModelStatus)
async def get_status():
    engine, config = get_shared_engine()
    return ModelStatus(
        loaded=engine is not None,
        backbone=config.get("backbone"),
        codec=config.get("codec"),
        device=config.get("device"),
        available_voices=list(VOICE_SAMPLES.keys()),
    )


@api_app.get("/voices")
async def list_voices():
    return {"voices": list(VOICE_SAMPLES.keys())}


@api_app.get("/backbones")
async def list_backbones():
    """List available backbone models."""
    return {
        "backbones": list(BACKBONE_CONFIGS.keys()),
        "configs": {
            name: {
                "repo": config["repo"],
                "supports_streaming": config.get("supports_streaming", False),
                "description": config.get("description", "")
            }
            for name, config in BACKBONE_CONFIGS.items()
        }
    }


@api_app.get("/codecs")
async def list_codecs():
    """List available codec models."""
    return {
        "codecs": list(CODEC_CONFIGS.keys()),
        "configs": {
            name: {
                "repo": config["repo"],
                "use_preencoded": config.get("use_preencoded", False),
                "description": config.get("description", "")
            }
            for name, config in CODEC_CONFIGS.items()
        }
    }


@api_app.post("/load-model", response_model=LoadModelResponse)
async def load_model_endpoint(request: LoadModelRequest):
    """
    Load or reload a model with specified parameters.
    
    If backbone or codec are not provided, they will be auto-selected based on available hardware.
    """
    try:
        result = load_model_with_params(
            backbone_choice=request.backbone,
            codec_choice=request.codec,
            device_choice=request.device
        )
        return LoadModelResponse(**result)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@api_app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """Synthesize speech from text. Returns audio file."""
    engine, _ = ensure_model_loaded()
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    start_time = time.time()
    
    try:
        ref_codes, ref_text = get_reference_data(request.voice)
        text_chunks = split_text_into_chunks(request.text.strip(), max_chars=MAX_CHARS_PER_CHUNK)
        
        all_audio = []
        sr = 24000
        silence_pad = np.zeros(int(sr * 0.15), dtype=np.float32)
        
        for i, chunk in enumerate(text_chunks):
            chunk_wav = engine.infer(chunk, ref_codes, ref_text)
            if chunk_wav is not None and len(chunk_wav) > 0:
                all_audio.append(chunk_wav)
                if i < len(text_chunks) - 1:
                    all_audio.append(silence_pad)
        
        if not all_audio:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        final_wav = np.concatenate(all_audio)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, final_wav, sr)
            output_path = tmp.name
        
        processing_time = time.time() - start_time
        duration = len(final_wav) / sr
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="tts_output.wav",
            headers={
                "X-Processing-Time": str(round(processing_time, 3)),
                "X-Audio-Duration": str(round(duration, 3)),
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@api_app.post("/synthesize/custom")
async def synthesize_custom(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    reference_text: str = Form(...),
):
    """Synthesize with custom voice. Upload audio + transcript."""
    engine, _ = ensure_model_loaded()
    
    start_time = time.time()
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            content = await reference_audio.read()
            tmp_audio.write(content)
            ref_audio_path = tmp_audio.name
        
        ref_codes = engine.encode_reference(ref_audio_path)
        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()
        
        text_chunks = split_text_into_chunks(text.strip(), max_chars=MAX_CHARS_PER_CHUNK)
        
        all_audio = []
        sr = 24000
        silence_pad = np.zeros(int(sr * 0.15), dtype=np.float32)
        
        for i, chunk in enumerate(text_chunks):
            chunk_wav = engine.infer(chunk, ref_codes, reference_text.strip())
            if chunk_wav is not None and len(chunk_wav) > 0:
                all_audio.append(chunk_wav)
                if i < len(text_chunks) - 1:
                    all_audio.append(silence_pad)
        
        os.unlink(ref_audio_path)
        
        if not all_audio:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        final_wav = np.concatenate(all_audio)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, final_wav, sr)
            output_path = tmp.name
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="tts_output.wav",
            headers={
                "X-Processing-Time": str(round(time.time() - start_time, 3)),
                "X-Audio-Duration": str(round(len(final_wav) / sr, 3)),
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- IMPORT AND MODIFY GRADIO APP ---
# We need to modify the gradio_app to share the model with API
import gradio_app as ga

# Patch the load_model function to share engine with API
_original_load_model = ga.load_model

def patched_load_model(backbone_choice, codec_choice, device_choice, enable_triton, max_batch_size):
    """Patched load_model that shares engine with API."""
    for result in _original_load_model(backbone_choice, codec_choice, device_choice, enable_triton, max_batch_size):
        yield result
    
    # After loading, share with API
    if ga.model_loaded and ga.tts is not None:
        device = "cuda" if torch.cuda.is_available() and device_choice != "CPU" else "cpu"
        set_shared_engine(ga.tts, {
            "backbone": backbone_choice,
            "codec": codec_choice,
            "device": device
        })
        print("Model shared with REST API")

ga.load_model = patched_load_model

# Get the Gradio demo
demo = ga.demo


# --- MOUNT API TO GRADIO ---
app = gr.mount_gradio_app(
    api_app,
    demo,
    path="/",
)

# The API will be available at /api/v1/...
# We need to create a sub-application for /api/v1
from fastapi import APIRouter
api_router = APIRouter()

# Re-mount all API routes under /api/v1
main_app = FastAPI(title="VieNeu-TTS Server")
main_app.mount("/api/v1", api_app)
main_app = gr.mount_gradio_app(main_app, demo, path="/")


if __name__ == "__main__":
    import uvicorn
    
    server_name = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", "7860"))
    
    print(f"\nVieNeu-TTS Server starting...")
    print(f"   Gradio UI: http://{server_name}:{server_port}")
    print(f"   REST API: http://{server_name}:{server_port}/api/v1")
    print(f"   API Docs: http://{server_name}:{server_port}/api/v1/docs\n")
    
    uvicorn.run(main_app, host=server_name, port=server_port)

