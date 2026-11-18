"""Turn two screen recordings (the web UI and Unity) into the README header GIF, side by side.

    uv run python scripts/compose_header.py webui.mp4 unity.mp4
    uv run python scripts/compose_header.py webui.mp4 unity.mp4 --left-start 3 --right-start 1.5 --seconds 12

Both recordings are trimmed to the same length, scaled to the same height, placed side by side and
encoded with one shared palette (ffmpeg's palettegen), which keeps a GIF small and free of banding.
Any format ffmpeg reads works (mp4, mov, mkv, webm). The ffmpeg used is the one bundled with
imageio-ffmpeg, so nothing else needs installing.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg


def ffmpeg(*args: str) -> None:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("left", help="the web UI recording")
    ap.add_argument("right", help="the Unity recording")
    ap.add_argument("--out", default="docs/header.gif")
    ap.add_argument("--left-start", type=float, default=0.0, help="seconds to skip at the start of the left one")
    ap.add_argument("--right-start", type=float, default=0.0, help="seconds to skip at the start of the right one")
    ap.add_argument("--seconds", type=float, default=None, help="length (default: the shorter recording)")
    ap.add_argument("--width", type=int, default=1280, help="width of the whole GIF")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed, e.g. 2 to fit a long run in")
    ap.add_argument("--colors", type=int, default=192, help="palette size; fewer makes a smaller file")
    a = ap.parse_args()

    half = a.width // 2 // 2 * 2  # each side, even for the encoder
    gap = 8
    duration = ["-t", str(a.seconds * a.speed)] if a.seconds else []
    side = (f"setpts=PTS/{a.speed},fps={a.fps},scale={half - gap // 2}:-2:flags=lanczos")
    pad = f"pad=iw+{gap // 2}:ih:color=0x16161c"
    graph = (f"[0:v]{side},{pad}[l];[1:v]{side},pad=iw+{gap // 2}:ih:{gap // 2}:0:color=0x16161c[r];"
             f"[l][r]hstack=inputs=2:shortest=1,pad=iw:ih+mod(ih\\,2):color=0x16161c[v]")
    inputs = ["-ss", str(a.left_start), *duration, "-i", a.left, "-ss", str(a.right_start), *duration, "-i", a.right]
    with tempfile.TemporaryDirectory() as tmp:
        palette = str(Path(tmp) / "palette.png")
        ffmpeg(*inputs, "-filter_complex", f"{graph};[v]palettegen=max_colors={a.colors}:stats_mode=diff", palette)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        ffmpeg(*inputs, "-i", palette, "-filter_complex",
               f"{graph};[v][2:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", "-loop", "0", a.out)
    size = Path(a.out).stat().st_size / 1e6
    note = "" if size <= 10 else "; over 10 MB: try --seconds, --fps 12, --width 1024 or --colors 128"
    print(f"{a.out}: {size:.1f} MB{note}")


if __name__ == "__main__":
    main()
