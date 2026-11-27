"""One clip from one motion prompt: start frame, video, tracks, fit, timing, glTF.

    <out>/<clip>/
      start.png, start_mask.png, camera.json, seeds.npz    the start frame and its seeds
      video.webm                                           the generated (or given) video
      tracks.npz                                           tracks, visibility, sampled depth
      motion.npz                                           the fitted motion, before looping
      <clip>.glb                                           the asset with the clip
      <clip>.wav                                           its sound, when the video has one
      capture.json                                         what was measured
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..animate import clip_name, split_prompt
from ..gltf_anim import AnimatedGLB
from . import clip as clip_
from . import motion as motion_
from .camera import render_start, seed_points, start_camera, write_start
from .fit import FitOptions, fit
from .rig import RigModel
from .video import DEFAULT_RECIPE, FPS, VideoSpec, generate, read_audio, read_frames, video_prompt, write_audio

LOOP_SECONDS = 4.2      # a loop's video: 107 frames, room for the ease out of rest, cycles, the ease back
ONCE_SECONDS = 2.2      # 56 frames


def prepare(asset: Path, folder: Path, spec: VideoSpec, points: int = 1500, device: str = "cpu"):
    """The start frame of ``asset`` at rest, its camera and seeds, written to ``folder``."""
    g = AnimatedGLB(asset)
    rig = RigModel(g, device=device)
    with torch.no_grad():
        rest = rig.pose_points(rig.vertex_basis(), *rig.zeros(1))[0].cpu().numpy()
    cam = start_camera(rest, spec.width, spec.height, yaw_deg=spec.view_yaw)
    rgb, mask, face, bary = render_start(rest, rig.faces, cam, g.uvs, g.texture)
    seeds = seed_points(mask, face, bary, points, rgb=rgb)
    write_start(folder, rgb, cam, seeds, mask)
    return g, rig, cam, seeds


def capture(asset: Path, prompt: str, folder: Path, spec: VideoSpec, subject: str | None = None, seed: int = 0,
            video: Path | None = None, device: str = "cuda", depth: bool = False, opts: FitOptions | None = None,
            log=print) -> dict:
    """Make (or take) a video for ``prompt`` and fit it onto ``asset``'s rig. Returns the clip's entry for
    clips.json: ``glb``, ``prompt``, ``loop``, ``loop_frames``, ``fps``, ``source``, ``events``,
    ``phases`` (one-shots) and ``sound`` when there is one."""
    from . import track

    text, loops = split_prompt(prompt)
    name = clip_name(text)
    folder = Path(folder)
    dev = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    g, rig, cam, seeds = prepare(asset, folder, spec, device=dev)
    log(f"[capture] {text!r}: start frame {cam.width}x{cam.height}, {len(seeds.pixels)} seeds")

    if video is None:
        seconds = LOOP_SECONDS if loops else ONCE_SECONDS
        video = generate(video_prompt(text, subject), folder / "start.png", folder / "start.png", seconds, spec,
                         seed, folder / "video.webm", log)
    frames, fps = read_frames(video, cam.width, cam.height)
    if abs(fps - FPS) > 0.5:
        log(f"[capture] note: the video runs at {fps:g} fps")
    tracks, visible = track.cotracker(frames, seeds, device=dev, log=log)
    observed = None
    if depth:
        from . import depth as depth_

        observed = depth_.sample(depth_.estimate(frames, device=dev, log=log), tracks)
    np.savez(folder / "tracks.npz", tracks=tracks, visible=visible,
             **({"depth": observed} if observed is not None else {}))

    opts = opts or FitOptions(anchor_end=True)
    result = fit(rig, seeds, tracks, visible, cam, opts, depth=observed, log=log)
    m = motion_.Motion(result.rotation, result.translation, result.travel)
    np.savez(folder / "motion.npz", rotation=m.rotation.cpu().numpy(), translation=m.translation.cpu().numpy(),
             travel=m.travel.cpu().numpy())

    audio = read_audio(video)
    sound_onsets = motion_.onsets(audio, 32000) if audio is not None else []
    entry = {"prompt": text, "loop": loops, "fps": fps, "source": "video", "video": str(video)}
    report = {"fit": result.report, "frames": len(frames)}
    if loops:
        m, cut = motion_.loop(rig, m, fps, skip_end=0.5 if opts.anchor_end else 0.0)
        entry["loop_frames"] = cut["frames"]
        report["loop"] = cut
        log(f"[capture] loop: {cut['frames']} frames ({cut['seconds']:.2f} s) from frame {cut['start']}, "
            f"seam {cut['seam']:.3f}")
        if audio is not None:
            a = int(cut["start"] / fps * 32000)
            audio = audio[a:a + int(cut["frames"] / fps * 32000)]
            sound_onsets = [round(o - cut["start"] / fps, 4) for o in sound_onsets
                            if 0 <= o - cut["start"] / fps < cut["seconds"]]
    else:
        entry["loop_frames"] = None
        entry["phases"] = report["phases"] = motion_.phases(rig, m, fps, sound_onsets[:8])
        log(f"[capture] phases: active {entry['phases']['active']} s, impact at {entry['phases']['impact']} s")
    found = motion_.contacts(rig, m, fps)
    events = motion_.footsteps(found)
    if "phases" in entry:
        events.append({"time": entry["phases"]["impact"], "type": "impact"})
    entry["events"] = sorted(events, key=lambda e: e["time"])
    report["contacts"] = found
    report["travel_speed"] = float(np.linalg.norm(m.travel.mean(0).cpu().numpy()) * fps)
    if audio is not None and len(audio):
        entry["sound"] = str(write_audio(folder / f"{name}.wav", audio))
        report["onsets"] = sound_onsets[:16]

    glb = clip_.write(asset, rig, m, fps, folder / f"{name}.glb", name=name,
                      extras={k: entry[k] for k in ("prompt", "loop", "fps", "source")})
    entry["glb"] = str(glb)
    (folder / "capture.json").write_text(json.dumps({**report, "entry": entry}, indent=2, default=float))
    return entry


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Motion from video: one clip from a motion prompt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="render the start frame (and its seeds) to make a video from elsewhere")
    s.add_argument("--asset", required=True, help="the rigged asset (2_rig/rigged.glb, or 3_motion/asset/<name>.glb)")
    s.add_argument("--out", required=True)
    s.add_argument("--size", default="640x640")
    s.add_argument("--view_yaw", type=float, default=VideoSpec.view_yaw,
                   help="where the camera stands, degrees round from the character's front (90: side on)")
    c = sub.add_parser("clip", help="generate (or take) a video and fit it onto the rig")
    c.add_argument("--asset", required=True)
    c.add_argument("--prompt", required=True, help='"An object jumps in place. [once]"; loops unless [once]')
    c.add_argument("--out", required=True)
    c.add_argument("--subject", default=None, help="what the character is, for the video prompt")
    c.add_argument("--video", default=None, help="use this video (made from the start frame) instead of generating")
    c.add_argument("--video_backend", default=None, choices=["sdcpp", "command"])
    c.add_argument("--video_url", default=None)
    c.add_argument("--video_model", default=None, help=f"sdcpp recipe (default {DEFAULT_RECIPE})")
    c.add_argument("--video_command", default=None)
    c.add_argument("--video_steps", type=int, default=None)
    c.add_argument("--size", default="640x640")
    c.add_argument("--view_yaw", type=float, default=None,
                   help=f"where the camera stands, degrees round from the front (default {VideoSpec.view_yaw:g})")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--device", default="cuda")
    c.add_argument("--depth", action="store_true",
                   help="also fit to Depth Anything's relative depth of the frames (made the benchmark worse)")
    a = ap.parse_args(argv)
    w, h = (int(v) for v in a.size.lower().split("x"))
    if a.cmd == "start":
        spec = VideoSpec(width=w, height=h, view_yaw=a.view_yaw)
        _, _, cam, seeds = prepare(Path(a.asset), Path(a.out), spec)
        print(f"[capture] start frame {Path(a.out) / 'start.png'} ({cam.width}x{cam.height}); make a video that "
              f"starts and ends on it, at {FPS} fps, then: assetgen capture clip --video <file> ...")
        return
    spec = VideoSpec.from_env(backend=a.video_backend, url=a.video_url, model=a.video_model,
                              command=a.video_command, steps=a.video_steps, width=w, height=h,
                              view_yaw=a.view_yaw)
    entry = capture(Path(a.asset), a.prompt, Path(a.out), spec, a.subject, a.seed,
                    Path(a.video) if a.video else None, a.device, depth=a.depth)
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
