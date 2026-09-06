from pathlib import Path
import argparse
import subprocess
import tempfile


def combine_audio(story_dir: Path):
    audio_dir = story_dir / "audio"
    output_dir = story_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "narration.wav"

    files = sorted(audio_dir.glob("*.wav"))

    if not files:
        raise SystemExit(f"Không tìm thấy file WAV trong: {audio_dir}")

    print("=" * 60)
    print(f"Story : {story_dir}")
    print(f"Files : {len(files)}")
    print(f"Output: {output_file}")
    print("=" * 60)

    # Tạo temporary concat list cho ffmpeg
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as f:
        concat_file = Path(f.name)

        for wav in files:
            # ffmpeg concat demuxer cần escape dấu '
            path = str(wav.resolve()).replace("'", "'\\''")
            f.write(f"file '{path}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_file),
        ]

        subprocess.run(cmd, check=True)

    finally:
        concat_file.unlink(missing_ok=True)

    print()
    print("=" * 60)
    print("DONE")
    print(f"Output: {output_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Combine story WAV segments into narration.wav"
    )

    parser.add_argument(
        "story",
        type=Path,
        help="Story directory, e.g. stories/truyen-001",
    )

    args = parser.parse_args()

    combine_audio(args.story)


if __name__ == "__main__":
    main()

