import re
import os
from typing import List
import numpy as np

# Pre-compiled regex patterns for splitting
PARA_SPLIT_RE = re.compile(r"[\r\n]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?\…])\s+")
SUBPART_SPLIT_RE = re.compile(r"(?<=[\,\;\:\-\–\—])\s+")

def join_audio_chunks(chunks: list[np.ndarray], sr: int, silence_p: float = 0.0, crossfade_p: float = 0.0) -> np.ndarray:
    """Join audio chunks with optional silence padding and crossfading."""
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]
    
    silence_samples = int(sr * silence_p)
    crossfade_samples = int(sr * crossfade_p)
    
    final_wav = chunks[0]
    
    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        
        if silence_samples > 0:
            silence = np.zeros(silence_samples, dtype=np.float32)
            final_wav = np.concatenate([final_wav, silence, next_chunk])
        elif crossfade_samples > 0:
            overlap = min(len(final_wav), len(next_chunk), crossfade_samples)
            if overlap > 0:
                fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                
                blended = (final_wav[-overlap:] * fade_out + next_chunk[:overlap] * fade_in)
                final_wav = np.concatenate([
                    final_wav[:-overlap],
                    blended,
                    next_chunk[overlap:]
                ])
            else:
                final_wav = np.concatenate([final_wav, next_chunk])
        else:
            final_wav = np.concatenate([final_wav, next_chunk])
            
    return final_wav

def split_text_into_chunks(text: str, max_chars: int = 256) -> List[str]:
    """
    Split raw text into chunks no longer than max_chars.
    """
    paragraphs = PARA_SPLIT_RE.split(text.strip())
    final_chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        sentences = SENTENCE_SPLIT_RE.split(para)
        
        buffer = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(sentence) > max_chars:
                if buffer:
                    final_chunks.append(buffer)
                    buffer = ""
                
                sub_parts = SUBPART_SPLIT_RE.split(sentence)
                for part in sub_parts:
                    part = part.strip()
                    if not part: continue
                    
                    if len(buffer) + 1 + len(part) <= max_chars:
                        buffer = (buffer + " " + part) if buffer else part
                    else:
                        if buffer: final_chunks.append(buffer)
                        buffer = part
                        
                        if len(buffer) > max_chars:
                            words = buffer.split()
                            current = ""
                            for word in words:
                                if current and len(current) + 1 + len(word) > max_chars:
                                    final_chunks.append(current)
                                    current = word
                                else:
                                    current = (current + " " + word) if current else word
                            buffer = current
            else:
                if buffer and len(buffer) + 1 + len(sentence) > max_chars:
                    final_chunks.append(buffer)
                    buffer = sentence
                else:
                    buffer = (buffer + " " + sentence) if buffer else sentence
        
        if buffer:
            final_chunks.append(buffer)
            buffer = ""

    return [c.strip() for c in final_chunks if c.strip()]

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")
