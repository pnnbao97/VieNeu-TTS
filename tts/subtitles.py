from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


# ============================================================
# Configuration
# ============================================================

DEFAULT_OUTPUT_NAME = "subtitles.srt"


# ============================================================
# Utilities
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def seconds_to_srt(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp:

        12.345 -> 00:00:12,345
    """
    total_ms = max(0, round(float(seconds) * 1000))

    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1_000)

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


def wrap_subtitle_text(text: str, max_chars: int = 42) -> str:
    """
    Light subtitle wrapping.

    We wrap only at whitespace and never change the actual words.
    Short sentences remain on one line.
    """
    text = re.sub(r"\s+", " ", text.strip())

    if len(text) <= max_chars:
        return text

    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        extra = len(word) if not current else len(word) + 1

        if current and current_len + extra > max_chars:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra

    if current:
        lines.append(" ".join(current))

    # SRT allows multiline subtitle blocks.
    return "\n".join(lines)


# ============================================================
# Subtitle generation
# ============================================================

def build_srt(
    timings: dict[str, Any],
    max_chars: int = 42,
) -> str:
    """
    Build SRT directly from output/timings.json.

    IMPORTANT:
    This function does NOT calculate timing from WAV duration.
    Each sentence's local start/end values come directly from
    timings.json. When multiple script files exist, their timelines
    are converted into one continuous global timeline by accumulating
    the previous script's final `end` value from timings.json.
    """
    blocks: list[str] = []
    subtitle_index = 1

    # timings.json stores each script's timeline starting at 0.
    # For the final narration.srt, convert each script's local timestamps
    # into one continuous global timeline by accumulating the previous
    # script's duration. The duration is derived ONLY from timings.json:
    # the last sentence's `end` is the end of that script's audio.
    global_offset = 0.0

    for segment_id in sorted(timings, key=lambda x: natural_key(Path(x))):
        sentences = timings[segment_id]

        if not isinstance(sentences, list):
            continue

        valid_sentences = []

        for item in sentences:
            text = str(item.get("text", "")).strip()

            if not text:
                continue

            start = float(item["start"])
            end = float(item["end"])

            if end <= start:
                print(
                    f"Warning: invalid timing "
                    f"{segment_id} #{item.get('index', '?')}: "
                    f"{start} -> {end}"
                )
                continue

            valid_sentences.append((item, text, start, end))

        for item, text, start, end in valid_sentences:
            # Convert local script time -> global narration time.
            global_start = global_offset + start
            global_end = global_offset + end

            subtitle_text = wrap_subtitle_text(
                text,
                max_chars=max_chars,
            )

            blocks.append(
                "\n".join(
                    [
                        str(subtitle_index),
                        f"{seconds_to_srt(global_start)} --> "
                        f"{seconds_to_srt(global_end)}",
                        subtitle_text,
                    ]
                )
            )

            subtitle_index += 1

        # Move the next script's timeline forward by this script's
        # duration. This is intentionally taken from timings.json,
        # not from WAV metadata.
        if valid_sentences:
            global_offset += max(end for _, _, _, end in valid_sentences)


    return "\n\n".join(blocks) + ("\n" if blocks else "")


def generate_subtitles(
    story_dir: Path,
    output_name: str = DEFAULT_OUTPUT_NAME,
    max_chars: int = 42,
) -> Path:
    output_dir = story_dir / "output"
    timings_path = output_dir / "timings.json"
    subtitles_path = output_dir / output_name

    if not timings_path.exists():
        raise FileNotFoundError(
            f"Missing timings.json: {timings_path}\n"
            "Run tts/generate.py first."
        )

    timings = load_json(timings_path)

    if not isinstance(timings, dict):
        raise ValueError(
            f"Invalid timings.json: expected an object, got "
            f"{type(timings).__name__}"
        )

    srt = build_srt(
        timings,
        max_chars=max_chars,
    )

    subtitles_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subtitles_path.write_text(
        srt,
        encoding="utf-8",
    )

    subtitle_count = sum(
        len(items)
        for items in timings.values()
        if isinstance(items, list)
    )

    print("=" * 60)
    print(f"Story     : {story_dir}")
    print(f"Source    : {timings_path}")
    print(f"Subtitles : {subtitles_path}")
    print(f"Entries   : {subtitle_count}")
    print("=" * 60)

    return subtitles_path


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate continuous SRT subtitles directly from "
            "VieNeu sentence-level timings.json."
        )
    )

    parser.add_argument(
        "story",
        type=Path,
        help="Story directory, e.g. stories/truyen-001",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help=(
            f"Output filename inside output/ "
            f"(default: {DEFAULT_OUTPUT_NAME})"
        ),
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=42,
        help="Maximum characters per subtitle line (default: 42).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_chars < 1:
        raise ValueError("--max-chars must be >= 1")

    generate_subtitles(
        story_dir=args.story,
        output_name=args.output,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
