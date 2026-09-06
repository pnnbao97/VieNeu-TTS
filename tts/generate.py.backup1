from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from vieneu import Vieneu
from vieneu.v3turbo import (
    DEFAULT_REP_WINDOW,
    gaps_to_silence,
    join_audio_chunks,
    normalize_to_chunks_v3_with_gaps,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_BATCH_SIZE = 8
CACHE_VERSION = 1
SENTENCE_GAP_SECONDS = 0.12

DEFAULT_STORY_ROOT = Path("stories")


# ============================================================
# Utilities
# ============================================================

def natural_key(path: Path):
    """
    Sort:
        001.txt
        002.txt
        010.txt
        011.txt

    instead of:
        001.txt
        010.txt
        011.txt
        002.txt
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def split_sentences(text: str) -> list[str]:
    """
    Split a script segment into sentences while preserving sentence text.

    Vietnamese narration is normally sentence-based, so sentence boundaries
    are used as the unit for accurate subtitle timing.
    """
    text = text.strip()
    if not text:
        return []

    text = re.sub(r"\s+", " ", text)

    sentences = re.split(r"(?<=[.!?…])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def combine_sentence_audio(
    sentence_audios: list[np.ndarray],
    sample_rate: int,
    gap_seconds: float = SENTENCE_GAP_SECONDS,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """
    Join sentence audio and return real sentence-level timestamps.

    The timestamps are measured from the actual generated audio duration,
    not estimated from text length or from the total segment duration.
    """
    if not sentence_audios:
        return np.array([], dtype=np.float32), []

    gap_samples = int(sample_rate * gap_seconds)
    silence = np.zeros(gap_samples, dtype=np.float32)

    parts: list[np.ndarray] = []
    timings: list[dict[str, float]] = []

    current_time = 0.0

    for index, audio in enumerate(sentence_audios):
        audio = np.asarray(audio, dtype=np.float32)

        duration = len(audio) / sample_rate
        start = current_time
        end = start + duration

        timings.append(
            {
                "index": index + 1,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
            }
        )

        parts.append(audio)
        current_time = end

        if index < len(sentence_audios) - 1:
            parts.append(silence)
            current_time += gap_seconds

    combined = np.concatenate(parts)

    return combined, timings


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def fingerprint_file(path: Path) -> str:
    """
    SHA-256 fingerprint of the reference audio.

    If reference.wav changes, the cached speaker embedding/reference
    codes will automatically be regenerated.
    """
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# Reference cache
# ============================================================

def cache_path_for_story(story_dir: Path) -> Path:
    return story_dir / "cache" / "reference.npz"


def load_reference_cache(
    cache_path: Path,
    reference_path: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load cached speaker_emb + ref_codes.

    Cache is valid only when:
        - cache version matches
        - reference.wav fingerprint matches
    """
    if not cache_path.exists():
        return None

    reference_fingerprint = fingerprint_file(reference_path)

    try:
        data = np.load(cache_path, allow_pickle=False)

        version = int(data["cache_version"])
        cached_fingerprint = str(data["reference_fingerprint"])

        if version != CACHE_VERSION:
            print("  Reference cache: INVALID (version mismatch)")
            return None

        if cached_fingerprint != reference_fingerprint:
            print("  Reference cache: INVALID (reference.wav changed)")
            return None

        speaker_emb = data["speaker_emb"]
        ref_codes = data["ref_codes"]

        return speaker_emb, ref_codes

    except Exception as exc:
        print(f"  Reference cache: INVALID ({exc})")
        return None


def save_reference_cache(
    cache_path: Path,
    reference_path: Path,
    speaker_emb: np.ndarray,
    ref_codes: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    reference_fingerprint = fingerprint_file(reference_path)

    np.savez(
        cache_path,
        cache_version=np.array(CACHE_VERSION, dtype=np.int64),
        reference_fingerprint=np.array(reference_fingerprint),
        speaker_emb=speaker_emb,
        ref_codes=ref_codes,
    )


def get_reference(
    tts: Vieneu,
    story_dir: Path,
    reference_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load reference from cache or encode it once.
    """
    cache_path = cache_path_for_story(story_dir)

    cached = load_reference_cache(
        cache_path,
        reference_path,
    )

    if cached is not None:
        speaker_emb, ref_codes = cached

        print("Reference cache: HIT")
        print(
            f"    speaker_emb: {speaker_emb.shape} "
            f"{speaker_emb.dtype}"
        )
        print(
            f"    ref_codes: {ref_codes.shape} "
            f"{ref_codes.dtype}"
        )

        return speaker_emb, ref_codes

    print("Reference cache: MISS")
    print("  Encoding reference.wav...")

    started = time.perf_counter()

    speaker_emb, ref_codes = tts.encode_reference(
        reference_path,
        denoise=True,
    )

    elapsed = time.perf_counter() - started

    print(f"  Reference encoded in {elapsed:.2f}s")
    print(
        f"    speaker_emb: {speaker_emb.shape} "
        f"{speaker_emb.dtype}"
    )
    print(
        f"    ref_codes: {ref_codes.shape} "
        f"{ref_codes.dtype}"
    )

    save_reference_cache(
        cache_path,
        reference_path,
        speaker_emb,
        ref_codes,
    )

    print(f"  Reference cache: SAVED -> {cache_path}")

    return speaker_emb, ref_codes


# ============================================================
# Audio generation
# ============================================================

def generate_single(
    tts: Vieneu,
    text: str,
    output_path: Path,
    speaker_emb: np.ndarray,
    ref_codes: np.ndarray,
    style: str,
) -> None:
    """
    Generate one text using already-encoded reference data.
    """
    audio = tts.engine.infer(
        text=text,
        speaker_emb=speaker_emb,
        ref_codes=ref_codes,
        style=style,
        use_ref_codes=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tts.save(audio, output_path)


def generate_batch_cached(
    tts: Vieneu,
    script_paths: list[Path],
    texts: list[str],
    output_paths: list[Path],
    speaker_emb: np.ndarray,
    ref_codes: np.ndarray,
    style: str,
    batch_size: int,
) -> dict[str, list[dict[str, Any]]]:
    """
    Generate multiple story segments while keeping sentence-level timing.

    Each .txt is split into sentences. Sentences are synthesized separately,
    joined back into their segment, and the real generated duration of each
    sentence is recorded.

    Returns:
        {
            "001": [
                {
                    "index": 1,
                    "text": "...",
                    "start": 0.0,
                    "end": 2.31,
                    "duration": 2.31
                },
                ...
            ]
        }
    """
    if not texts:
        return {}

    sampling = dict(
        temperature=0.8,
        top_k=25,
        top_p=0.95,
        max_new_frames=300,
        repetition_penalty=1.2,
        repetition_window=DEFAULT_REP_WINDOW,
    )

    # --------------------------------------------------------
    # Split every segment into sentences
    # --------------------------------------------------------
    segment_sentences: list[list[str]] = []
    flat_sentences: list[str] = []
    sentence_owner: list[int] = []

    for segment_index, text in enumerate(texts):
        sentences = split_sentences(text)
        segment_sentences.append(sentences)

        for sentence in sentences:
            flat_sentences.append(sentence)
            sentence_owner.append(segment_index)

    if not flat_sentences:
        empty = np.array([], dtype=np.float32)

        for output_path in output_paths:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tts.save(empty, output_path)

        return {
            path.stem: []
            for path in script_paths
        }

    print(
        f"  Sentences: {len(flat_sentences)} "
        f"across {len(texts)} segment(s)"
    )

    # --------------------------------------------------------
    # Normalize every sentence into VieNeu chunks.
    #
    # Chunk ownership is tracked so we can reconstruct:
    #
    #     chunks -> sentence -> segment
    #
    # while keeping the real duration of each sentence.
    # --------------------------------------------------------
    per_sentence_gaps: list[Any] = []
    flat_chunks: list[str] = []
    chunk_owner: list[int] = []

    for sentence_index, sentence in enumerate(flat_sentences):
        chunks, gaps = normalize_to_chunks_v3_with_gaps(
            sentence,
            max_chars=256,
        )

        per_sentence_gaps.append(gaps)

        for chunk in chunks:
            flat_chunks.append(chunk)
            chunk_owner.append(sentence_index)

    empty = np.array([], dtype=np.float32)

    if not flat_chunks:
        return {
            path.stem: []
            for path in script_paths
        }

    # --------------------------------------------------------
    # Direct cached-reference inference
    # --------------------------------------------------------
    flat_wavs = tts._infer_chunks(
        flat_chunks,
        speaker_emb,
        ref_codes,
        True,
        max(1, int(batch_size)),
        sampling,
    )

    # --------------------------------------------------------
    # Group generated chunks back to sentences
    # --------------------------------------------------------
    sentence_chunks: list[list[np.ndarray]] = [
        []
        for _ in flat_sentences
    ]

    for wav, sentence_index in zip(flat_wavs, chunk_owner):
        sentence_chunks[sentence_index].append(wav)

    # --------------------------------------------------------
    # Join chunks belonging to each sentence.
    #
    # IMPORTANT:
    # Do NOT watermark here. Watermark is applied exactly once
    # after the whole segment has been assembled.
    # --------------------------------------------------------
    sentence_audios: list[np.ndarray] = []

    for sentence_index, chunks in enumerate(sentence_chunks):
        if not chunks:
            audio = empty
        else:
            audio = join_audio_chunks(
                chunks,
                tts.sample_rate,
                silence_ps=gaps_to_silence(
                    per_sentence_gaps[sentence_index]
                ),
            )

        sentence_audios.append(audio)

    # --------------------------------------------------------
    # Reconstruct each segment and save real sentence timings.
    # --------------------------------------------------------
    segment_metadata: dict[str, list[dict[str, Any]]] = {}
    sentence_cursor = 0

    for script_path, sentences, output_path in zip(
        script_paths,
        segment_sentences,
        output_paths,
    ):
        audios = sentence_audios[
            sentence_cursor:
            sentence_cursor + len(sentences)
        ]

        sentence_cursor += len(sentences)

        combined_audio, timings = combine_sentence_audio(
            audios,
            tts.sample_rate,
        )

        # Watermark exactly once per final segment.
        combined_audio = tts._apply_watermark(combined_audio)

        metadata_items: list[dict[str, Any]] = []

        for sentence, timing in zip(sentences, timings):
            metadata_items.append(
                {
                    "index": timing["index"],
                    "text": sentence,
                    "start": timing["start"],
                    "end": timing["end"],
                    "duration": timing["duration"],
                }
            )

        segment_metadata[script_path.stem] = metadata_items

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tts.save(
            combined_audio,
            output_path,
        )

    return segment_metadata


# ============================================================
# Story generation
# ============================================================

def generate_story(
    story_dir: Path,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    started_total = time.perf_counter()

    config_path = story_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config.json: {config_path}"
        )

    config = load_json(config_path)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    voice_path = story_dir / config.get(
        "voice",
        "voice/reference.wav",
    )

    script_dir = story_dir / "script"
    audio_dir = story_dir / "audio"
    output_dir = story_dir / "output"
    timings_path = output_dir / "timings.json"

    if not voice_path.exists():
        raise FileNotFoundError(
            f"Missing reference voice: {voice_path}"
        )

    if not script_dir.exists():
        raise FileNotFoundError(
            f"Missing script directory: {script_dir}"
        )

    # --------------------------------------------------------
    # Style
    # --------------------------------------------------------
    style = config.get(
        "style",
        "doc_truyen",
    )

    # --------------------------------------------------------
    # Script files
    # --------------------------------------------------------
    script_files = sorted(
        script_dir.glob("*.txt"),
        key=natural_key,
    )

    if not script_files:
        print(f"No .txt files found in {script_dir}")
        return

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------
    generated = 0
    skipped = 0
    failed = 0

    # Keep existing timings so a later run with skipped files
    # never destroys their metadata.
    all_metadata: dict[str, list[dict[str, Any]]] = {}

    if timings_path.exists():
        try:
            existing_metadata = load_json(timings_path)

            if isinstance(existing_metadata, dict):
                all_metadata.update(existing_metadata)

                print(
                    f"Timings cache: HIT -> {timings_path}"
                )

        except Exception as exc:
            print(
                f"Timings cache: INVALID ({exc})"
            )

    # --------------------------------------------------------
    # Initialize model
    # --------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Story: {story_dir}")
    print("=" * 60)

    print(f"Voice : {voice_path}")
    print(f"Style : {style}")
    print(f"Batch : {batch_size}")
    print(f"Texts : {len(script_files)}")
    print()

    tts = Vieneu(
        mode="v3turbo",
        device="auto",
        backend="auto",
        threads=0,
        max_batch_size=max(batch_size, 1),
    )

    try:
        # ----------------------------------------------------
        # Load / create reference cache
        # ----------------------------------------------------
        speaker_emb, ref_codes = get_reference(
            tts,
            story_dir,
            voice_path,
        )

        print()

        # ----------------------------------------------------
        # Determine which files actually need generation
        # ----------------------------------------------------
        pending_scripts: list[Path] = []
        pending_texts: list[str] = []
        pending_outputs: list[Path] = []

        for script_path in script_files:
            output_path = (
                audio_dir /
                f"{script_path.stem}.wav"
            )

            if output_path.exists() and not force:
                skipped += 1

                print(
                    f"[SKIP] {script_path.name} "
                    f"-> {output_path.name}"
                )

                continue

            text = read_text(script_path)

            if not text:
                failed += 1

                print(
                    f"[FAIL] {script_path.name}: "
                    f"empty text"
                )

                continue

            pending_scripts.append(script_path)
            pending_texts.append(text)
            pending_outputs.append(output_path)

        if not pending_texts:
            # Persist existing metadata even when everything is skipped.
            output_dir.mkdir(parents=True, exist_ok=True)
            timings_path.write_text(
                json.dumps(
                    all_metadata,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            elapsed = time.perf_counter() - started_total

            print()
            print("=" * 60)
            print("Nothing to generate.")
            print(f"Skipped : {skipped}")
            print(f"Timings : {timings_path}")
            print(f"Time    : {elapsed:.2f}s")
            print("=" * 60)

            return

        # ----------------------------------------------------
        # Generate in batches
        # ----------------------------------------------------
        print(
            f"Generating {len(pending_texts)} segment(s)..."
        )
        print()

        for start in range(
            0,
            len(pending_texts),
            max(1, batch_size),
        ):
            end = min(
                start + max(1, batch_size),
                len(pending_texts),
            )

            batch_scripts = pending_scripts[start:end]
            batch_texts = pending_texts[start:end]
            batch_outputs = pending_outputs[start:end]

            print(
                f"[BATCH] "
                f"{start + 1}-{end} / "
                f"{len(pending_texts)}"
            )

            batch_started = time.perf_counter()

            try:
                batch_metadata = generate_batch_cached(
                    tts=tts,
                    script_paths=batch_scripts,
                    texts=batch_texts,
                    output_paths=batch_outputs,
                    speaker_emb=speaker_emb,
                    ref_codes=ref_codes,
                    style=style,
                    batch_size=batch_size,
                )

                all_metadata.update(batch_metadata)
                generated += len(batch_texts)

                batch_elapsed = (
                    time.perf_counter() -
                    batch_started
                )

                print(
                    f"  OK: {len(batch_texts)} segment(s) "
                    f"in {batch_elapsed:.2f}s"
                )

            except Exception as exc:
                # ------------------------------------------------
                # Robust fallback:
                # retry each segment through the SAME
                # sentence-aware pipeline with batch_size=1.
                # This guarantees timings.json remains complete.
                # ------------------------------------------------
                print()
                print(
                    f"  Batch failed: {exc}"
                )
                print(
                    "  Falling back to individual generation..."
                )

                for script_path, text, output_path in zip(
                    batch_scripts,
                    batch_texts,
                    batch_outputs,
                ):
                    try:
                        single_started = time.perf_counter()

                        single_metadata = generate_batch_cached(
                            tts=tts,
                            script_paths=[script_path],
                            texts=[text],
                            output_paths=[output_path],
                            speaker_emb=speaker_emb,
                            ref_codes=ref_codes,
                            style=style,
                            batch_size=1,
                        )

                        all_metadata.update(
                            single_metadata
                        )

                        generated += 1

                        single_elapsed = (
                            time.perf_counter() -
                            single_started
                        )

                        print(
                            f"  [OK] {script_path.name} "
                            f"({single_elapsed:.2f}s)"
                        )

                    except Exception as single_exc:
                        failed += 1

                        print(
                            f"  [FAIL] {script_path.name}: "
                            f"{single_exc}"
                        )

                print()

            # Persist after every batch. If a later batch fails or the
            # process is interrupted, completed timings are not lost.
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            timings_path.write_text(
                json.dumps(
                    all_metadata,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print()

    finally:
        tts.close()

    # --------------------------------------------------------
    # Final timings persistence
    # --------------------------------------------------------
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timings_path.write_text(
        json.dumps(
            all_metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    elapsed = time.perf_counter() - started_total

    print("=" * 60)
    print(f"Generated : {generated}")
    print(f"Failed    : {failed}")
    print(f"Skipped   : {skipped}")
    print(f"Timings   : {timings_path}")
    print(f"Time      : {elapsed:.2f}s")
    print("=" * 60)


# ============================================================
# Story discovery
# ============================================================

def discover_stories(
    stories_root: Path,
) -> list[Path]:

    if not stories_root.exists():
        raise FileNotFoundError(
            f"Stories directory not found: {stories_root}"
        )

    stories = [
        path
        for path in stories_root.iterdir()
        if path.is_dir()
        and (path / "config.json").exists()
    ]

    return sorted(
        stories,
        key=natural_key,
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Vietnamese narration using VieNeu-TTS."
    )

    parser.add_argument(
        "story",
        nargs="?",
        type=Path,
        help="Story directory, e.g. stories/truyen-001",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all stories under stories/",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing WAV files.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE}).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.all and args.story:
        raise ValueError(
            "Use either STORY or --all, not both."
        )

    if not args.all and not args.story:
        raise ValueError(
            "Specify a story directory or use --all."
        )

    if args.all:
        stories = discover_stories(
            DEFAULT_STORY_ROOT
        )

        if not stories:
            print(
                f"No stories found in "
                f"{DEFAULT_STORY_ROOT}"
            )

            return

        for story_dir in stories:
            generate_story(
                story_dir=story_dir,
                force=args.force,
                batch_size=args.batch_size,
            )

    else:
        generate_story(
            story_dir=args.story,
            force=args.force,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()