import numpy as np
import re
import logging
from typing import List, Dict, Optional, Any

# Configure logging
logger = logging.getLogger("Vieneu.Utils")

# Persistent cache for weights to avoid recomputing if frame_length is constant
_WEIGHT_CACHE: Dict[int, np.ndarray] = {}

def normalize_device(device: str) -> str:
    """
    Standardize device strings across backends.
    Maps 'gpu', 'cuda:*' to 'cuda', handles 'mps', 'xpu', and defaults to 'cpu'.
    """
    d = device.lower()
    if "cuda" in d or d == "gpu":
        return "cuda"
    if d == "mps":
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if d == "xpu":
        return "xpu"
    return "cpu"

def _linear_overlap_add(frames: List[np.ndarray], stride: int) -> np.ndarray:
    """
    Perform linear overlap-add on a list of audio frames.

    Original implementation inspired by:
    https://github.com/facebookresearch/encodec/blob/main/encodec/utils.py

    Args:
        frames: List of audio frames to join.
        stride: Stride between frames in samples.

    Returns:
        Joined audio waveform.
    """
    if not frames:
        return np.array([], dtype=np.float32)

    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]

    total_size = 0
    for i, frame in enumerate(frames):
        frame_end = stride * i + frame.shape[-1]
        total_size = max(total_size, frame_end)

    sum_weight = np.zeros(total_size, dtype=dtype)
    out = np.zeros((*shape, total_size), dtype=dtype)

    offset: int = 0
    for frame in frames:
        frame_length = frame.shape[-1]

        if frame_length not in _WEIGHT_CACHE or _WEIGHT_CACHE[frame_length].dtype != dtype:
            # Recompute weight if not in cache or dtype mismatch
            t = np.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
            weight = np.abs(0.5 - (t - 0.5))
            _WEIGHT_CACHE[frame_length] = weight
        else:
            weight = _WEIGHT_CACHE[frame_length]

        out[..., offset : offset + frame_length] += weight * frame
        sum_weight[offset : offset + frame_length] += weight
        offset += stride

    # Ensure no division by zero; use small epsilon if needed
    safe_sum_weight = np.where(sum_weight > 0, sum_weight, 1.0)
    return out / safe_sum_weight

def _compile_codec_with_triton(codec: Any) -> bool:
    """
    Compile codec with Triton for faster decoding (Windows/Linux compatible).

    Args:
        codec: The codec model to compile.

    Returns:
        True if compilation was successful, False otherwise.
    """
    try:
        import triton
        import torch

        if hasattr(codec, 'dec') and hasattr(codec.dec, 'resblocks'):
            if len(codec.dec.resblocks) > 2:
                # Use torch.compile with triton-friendly backend
                codec.dec.resblocks[2].forward = torch.compile(
                    codec.dec.resblocks[2].forward,
                    mode="reduce-overhead",
                    dynamic=True
                )
                logger.info("   ✅ Triton compilation enabled for codec")
        return True

    except ImportError:
        # Silently fail for optional triton optimization
        return False
    except Exception as e:
        logger.error(f"   ⚠️ Triton compilation failed: {e}")
        return False

# Pre-compile regex for speech token extraction
RE_SPEECH_TOKEN = re.compile(r"<\|speech_(\d+)\|>")

def extract_speech_ids(codes_str: str) -> List[int]:
    """Extract speech token IDs from a string using regex."""
    return [int(num) for num in RE_SPEECH_TOKEN.findall(codes_str)]
