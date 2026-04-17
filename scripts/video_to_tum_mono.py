#!/usr/bin/env python3
"""Convert a video into a TUM-style monocular dataset.

Example:
    python3 scripts/video_to_tum_mono.py /path/to/konkuk_lake.mp4
    python3 scripts/video_to_tum_mono.py /path/to/konkuk_lake.mp4 --width 640 --height 480

This creates:
    data/<video_stem>_tum_mono/
      ├── rgb/
      │   ├── 0.000000.png
      │   ├── 0.100000.png
      │   └── ...
      └── rgb.txt
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

RGB_TXT_HEADER = """# color images
# file: 'rgb.txt'
# timestamp filename
"""
TIMESTAMP_PRECISION = Decimal("0.000001")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a video into a TUM monocular dataset (rgb/ + rgb.txt)."
    )
    parser.add_argument("input_video", type=Path, help="Input video file path")
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        help="Output dataset directory (default: data/<video_stem>_tum_mono)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Sampling rate in frames per second (default: 10)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Optional clip start time in seconds (default: 0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional clip duration in seconds",
    )
    parser.add_argument(
        "--image-ext",
        choices=("png", "jpg"),
        default="png",
        help="Extracted image format (default: png)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional output image width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional output image height in pixels",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable to use (default: ffmpeg)",
    )
    return parser.parse_args()


def require_executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise SystemExit(f"Required executable not found in PATH: {name}")
    return resolved


def quantized_timestamp(frame_index: int, fps: Decimal) -> str:
    timestamp = (Decimal(frame_index) / fps).quantize(
        TIMESTAMP_PRECISION, rounding=ROUND_HALF_UP
    )
    return format(timestamp, ".6f")


def build_output_dir(input_video: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return Path("data") / f"{input_video.stem}_tum_mono"


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise SystemExit(
                f"Output directory already exists: {output_dir}\n"
                "Use --force to overwrite it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def build_filter_chain(fps: float, width: int | None, height: int | None) -> str:
    filters = [f"fps={fps}"]

    if width is None and height is None:
        return ",".join(filters)
    if width is None or height is None:
        raise SystemExit("--width and --height must be provided together")
    if width <= 0 or height <= 0:
        raise SystemExit("--width and --height must be greater than 0")

    filters.append(f"scale={width}:{height}")
    return ",".join(filters)


def run_ffmpeg(
    ffmpeg_bin: str,
    input_video: Path,
    temp_frames_dir: Path,
    fps: float,
    width: int | None,
    height: int | None,
    start_time: float,
    duration: float | None,
    image_ext: str,
) -> None:
    output_pattern = temp_frames_dir / f"frame_%06d.{image_ext}"
    filter_chain = build_filter_chain(fps=fps, width=width, height=height)
    cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]

    if start_time > 0:
        cmd.extend(["-ss", str(start_time)])

    cmd.extend(["-i", str(input_video)])

    if duration is not None:
        cmd.extend(["-t", str(duration)])

    if image_ext == "jpg":
        cmd.extend(["-q:v", "2"])

    cmd.extend(["-vf", filter_chain, "-start_number", "0", str(output_pattern)])

    subprocess.run(cmd, check=True)


def move_frames_and_write_index(
    temp_frames_dir: Path,
    output_dir: Path,
    fps_value: float,
    image_ext: str,
) -> int:
    rgb_dir = output_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(temp_frames_dir.glob(f"frame_*.{image_ext}"))
    if not frames:
        raise SystemExit("No frames were extracted. Check the video path or sampling options.")

    fps = Decimal(str(fps_value))
    if fps <= 0:
        raise SystemExit("--fps must be greater than 0")

    rgb_txt_path = output_dir / "rgb.txt"
    with rgb_txt_path.open("w", encoding="utf-8") as rgb_txt:
        rgb_txt.write(RGB_TXT_HEADER)
        for index, frame_path in enumerate(frames):
            timestamp = quantized_timestamp(index, fps)
            filename = f"{timestamp}.{image_ext}"
            target_path = rgb_dir / filename
            shutil.move(str(frame_path), str(target_path))
            rgb_txt.write(f"{timestamp} rgb/{filename}\n")

    return len(frames)


def main() -> int:
    args = parse_args()

    if args.fps <= 0:
        raise SystemExit("--fps must be greater than 0")
    if args.start_time < 0:
        raise SystemExit("--start-time must be >= 0")
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be greater than 0")
    if (args.width is None) != (args.height is None):
        raise SystemExit("--width and --height must be provided together")
    if args.width is not None and args.width <= 0:
        raise SystemExit("--width must be greater than 0")
    if args.height is not None and args.height <= 0:
        raise SystemExit("--height must be greater than 0")
    if not args.input_video.is_file():
        raise SystemExit(f"Input video not found: {args.input_video}")

    ffmpeg_bin = require_executable(args.ffmpeg_bin)
    output_dir = build_output_dir(args.input_video, args.output_dir)
    prepare_output_dir(output_dir, args.force)

    with tempfile.TemporaryDirectory(prefix="video_to_tum_mono_") as temp_dir:
        temp_frames_dir = Path(temp_dir)
        run_ffmpeg(
            ffmpeg_bin=ffmpeg_bin,
            input_video=args.input_video,
            temp_frames_dir=temp_frames_dir,
            fps=args.fps,
            width=args.width,
            height=args.height,
            start_time=args.start_time,
            duration=args.duration,
            image_ext=args.image_ext,
        )
        frame_count = move_frames_and_write_index(
            temp_frames_dir=temp_frames_dir,
            output_dir=output_dir,
            fps_value=args.fps,
            image_ext=args.image_ext,
        )

    print(f"Created {frame_count} frames in {output_dir}")
    print(f"RGB index: {output_dir / 'rgb.txt'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
