"""Render the README header: a run's input picture, then its character playing each clip, captioned.

    uv run python scripts/make_header.py outputs/fox --picture outputs/fox/input/fox.png --out docs/header.gif
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from assetgen.gltf_anim import AnimatedGLB
from assetgen.preview import render_mesh
from assetgen.splats import make_camera

W, H, FPS, SIZE, SUPERSAMPLE = 1280, 400, 15, 400, 2
BG = (22, 22, 28)
MUTED, BODY, TITLE, ACCENT = (140, 140, 156), (170, 170, 184), (232, 232, 239), (240, 170, 80)


def load_font(size: int, bold: bool = False):
    for name in (("segoeuib.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def clip_order(character: dict) -> list[str]:
    """Locomotion from slow to fast, then the other loops, then the one-shots."""
    blend = [p["clip"] for p in character["locomotion"]["blend"]]
    rest = [c["name"] for c in character["clips"] if c["name"] not in blend]
    return blend + sorted(rest, key=lambda n: next(c["loop"] for c in character["clips"] if c["name"] == n),
                          reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", help="a finished run folder (with <name>.glb and <name>.assetgen)")
    ap.add_argument("--picture", required=True, help="the picture the character was made from")
    ap.add_argument("--out", default="docs/header.gif")
    ap.add_argument("--loop-seconds", type=float, default=2.4, help="how long each looping clip plays")
    a = ap.parse_args()

    run = Path(a.run)
    glb = next(run.glob("*.glb"))
    character = json.loads(glb.with_suffix(".assetgen").read_text())
    clips = {c["name"]: c for c in character["clips"]}
    g = AnimatedGLB(glb)
    index = {name: i for i, name in enumerate(g.animations)}
    order = clip_order(character)

    # One camera for every clip, framing all of their poses, so the character keeps its size.
    poses = np.concatenate([g.posed(index[n], g.duration(index[n]) * k / 8) for n in order for k in range(9)])
    lo, hi = poses.min(0), poses.max(0)
    center = torch.tensor((lo + hi) / 2, dtype=torch.float32)
    radius = 0.62 * float(np.linalg.norm(hi - lo)) / 2 / math.tan(math.radians(15))
    yaw = math.radians(40)
    eye = center + radius * torch.tensor([math.sin(yaw) * 0.96, 0.28, math.cos(yaw) * 0.96])
    cam = make_camera(eye, center, 30.0, SIZE * SUPERSAMPLE)

    picture = np.asarray(Image.open(a.picture).convert("RGB").resize((250, 250), Image.LANCZOS)).copy()
    picture[picture.sum(2) < 24] = BG  # a black background blends into the banner
    picture = Image.fromarray(picture)

    def frame(fox: Image.Image, clip: dict) -> Image.Image:
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        im.paste(picture, (60, 60))
        d.text((185, 335), "one picture in", font=load_font(18), fill=MUTED, anchor="mm")
        d.text((365, 190), "→", font=load_font(64), fill=(90, 90, 104), anchor="mm")
        im.paste(fox, (405, 0))
        d.text((605, 372), "rigged and animated, out", font=load_font(18), fill=MUTED, anchor="mm")
        d.text((860, 92), "assetgen", font=load_font(40, bold=True), fill=TITLE)
        d.text((860, 146), f"“{clip['prompt']}”", font=load_font(21), fill=ACCENT)
        speed = f" · {clip['speed']:.2f} m/s" if clip.get("speed", 0) > 0.05 else ""
        d.text((860, 182), ("loops seamlessly" if clip["loop"] else "plays once") + speed, font=load_font(17),
               fill=MUTED)
        d.text((860, 262), "a sentence or a picture in;", font=load_font(17), fill=BODY)
        d.text((860, 286), "a character for Unity and Godot out", font=load_font(17), fill=BODY)
        return im

    frames = []
    for name in order:
        i, clip = index[name], clips[name]
        seconds = a.loop_seconds if clip["loop"] else g.duration(i)
        for k in range(int(seconds * FPS)):
            t = (k / FPS) % g.duration(i)
            rgb = render_mesh(g.posed(i, t), g.faces, cam, g.uvs, g.texture, background=tuple(c / 255 for c in BG))[0]
            fox = Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.LANCZOS)
            frames.append(frame(fox, clip))

    palette = frames[len(frames) // 3].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(a.out, save_all=True, append_images=quantized[1:], duration=int(1000 / FPS), loop=0,
                      optimize=True)
    print(f"{a.out}: {len(frames)} frames, clips {', '.join(order)}")


if __name__ == "__main__":
    main()
