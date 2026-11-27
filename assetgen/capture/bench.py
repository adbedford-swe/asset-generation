"""Score motion capture on animations whose truth is known.

A known animation (the Khronos Fox's walk, a clip UniMate made) is played from the rest pose, the way
a generated video starts from the start frame: rest, a short blend into the animation, then the
animation for a few seconds. Its seeds are followed exactly (``--tracker exact``, which isolates the
fit) or by CoTracker on a rendered video (``--tracker cotracker``, the whole method minus the video
model). The fitted motion is compared with the truth in 3D:

- vertex error: mean distance between fitted and true vertices, as a share of the asset's size;
- the same seen only across the camera's view (what one camera can see) and along it (depth);
- the rest pose scored the same way, as the error of not moving at all.

    python -m assetgen.capture.bench --glb Fox.glb --anim Walk --out bench/fox_walk
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..gltf_anim import AnimatedGLB
from . import track
from .camera import project, render_start, seed_points, start_camera, write_start
from .fit import FitOptions, Objective, fit
from .rig import RigModel, from_animation


def played_from_rest(rig: RigModel, g: AnimatedGLB, anim: int, fps: float = 24.0, seconds: float = 3.0,
                     blend: float = 0.5, end_at_rest: bool = False,
                     follow: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Rest, a ``blend``-second ease into ``anim``, then ``anim`` repeated to ``seconds`` in all; with
    ``end_at_rest``, an ease back to rest at the end (to where it started, as a video that ends on its
    start frame does). Each repeat carries on from where the last one got to, so a walk covers ground;
    with ``follow`` the camera follows it instead, as a video model's camera often does, so it walks on
    the spot in the picture while its planted feet slide back at its speed (see :func:`travel_speed`)."""
    duration = g.duration(anim)
    n_blend = max(1, int(round(blend * fps)))
    n_total = int(round(seconds * fps)) + 1
    n_anim = n_total - n_blend - (n_blend if end_at_rest else 0)
    elapsed = np.arange(n_anim) / fps
    rot, tr = from_animation(rig, g, anim, elapsed % duration)
    _, ends = from_animation(rig, g, anim, [0.0, duration])
    # Whole cycles completed carry the hub on; following, the camera takes away the ground covered so far.
    covered = np.floor(elapsed / duration) - (elapsed / duration if follow else 0.0)
    tr = tr + torch.as_tensor(covered, dtype=tr.dtype)[:, None] * (ends[1] - ends[0])[None]
    ease = torch.as_tensor(0.5 - 0.5 * np.cos(np.linspace(0, math.pi, n_blend + 1)[:-1]), dtype=rot.dtype)
    head_r, head_t = ease[:, None, None] * rot[:1], ease[:, None] * tr[:1]
    parts_r, parts_t = [head_r, rot], [head_t, tr]
    if end_at_rest:
        parts_r.append(ease.flip(0)[:, None, None] * rot[-1:])
        parts_t.append(ease.flip(0)[:, None] * tr[-1:])
    return torch.cat(parts_r), torch.cat(parts_t)


def travel_speed(rig: RigModel, g: AnimatedGLB, anim: int) -> float:
    """The ground speed of ``anim`` (world units a second): its hub's horizontal travel over one cycle."""
    from .motion import hub_parent_rotation

    _, ends = from_animation(rig, g, anim, [0.0, g.duration(anim)])
    with torch.no_grad():
        d = hub_parent_rotation(rig).to(ends) @ (ends[1] - ends[0])
    return float(torch.hypot(d[0], d[2])) / g.duration(anim)


def contact_speed(rig: RigModel, rotation, translation, fps: float, band: float = 0.015) -> float | None:
    """The speed (world units a second) at which the parts of a motion resting on the ground slide along
    it: the speed a motion authored in place, or filmed by a following camera, travels at."""
    with torch.no_grad():
        P = rig.pose_points(rig.vertex_basis(), rotation, translation).cpu().double().numpy()
    height = P[..., 1] - P[0, :, 1].min()
    v = np.linalg.norm(np.diff(P[..., [0, 2]], axis=0), axis=-1) * fps
    low = height[:-1] < band * rig.size
    speeds = [float(np.median(v[t][low[t]])) for t in range(len(v)) if low[t].sum() > 5]
    return float(np.median(speeds)) if speeds else None


def render_video(rig: RigModel, rotation, translation, cam, uvs, texture) -> np.ndarray:
    from ..preview import render_mesh

    with torch.no_grad():
        verts = rig.pose_points(rig.vertex_basis(), rotation, translation).cpu().numpy()
    light = (-cam.view[0, :3] + 1.5 * torch.tensor([0.0, 1.0, 0.0]) - 0.5 * cam.view[2, :3]).tolist()
    return np.stack([(render_mesh(v, rig.faces, cam, uvs, texture, light=light)[0].numpy() * 255).round()
                     .astype(np.uint8) for v in verts])


def score(rig: RigModel, fitted, truth, cam) -> dict:
    """Vertex errors of ``fitted`` against ``truth`` (each a (rotation, translation) pair), split into the
    part across the camera's view and the part along it."""
    B = rig.vertex_basis()
    with torch.no_grad():
        P = rig.pose_points(B, *fitted).cpu().numpy()
        G = rig.pose_points(B, *truth).cpu().numpy()
        R0 = rig.pose_points(B, *rig.zeros(len(G))).cpu().numpy()
    axis = cam.view[2, :3].double().numpy()  # the camera's viewing direction in world space

    def errors(X):
        e = X - G
        along = np.abs(e @ axis)
        across = np.linalg.norm(e - (e @ axis)[..., None] * axis, axis=-1)
        total = np.linalg.norm(e, axis=-1)
        per_frame = total.mean(1) / rig.size
        return {"vertex_mean": float(total.mean() / rig.size), "vertex_p95": float(np.percentile(total, 95) / rig.size),
                "across_view": float(across.mean() / rig.size), "along_view": float(along.mean() / rig.size),
                "worst_frame": float(per_frame.max())}

    return {"fit": errors(P), "rest": errors(R0)}


def contact_sheet(rig: RigModel, fitted, truth, cam_side, frames: int = 6, size: int = 256) -> np.ndarray:
    """Truth (top) and fit (bottom) seen from the side, where depth errors show."""
    from ..preview import render_mesh

    B = rig.vertex_basis()
    T = len(truth[0])
    pick = np.linspace(0, T - 1, frames).round().astype(int)
    rows = []
    with torch.no_grad():
        for params, colour in ((truth, (0.55, 0.75, 0.95)), (fitted, (0.95, 0.7, 0.5))):
            V = rig.pose_points(B, params[0][pick], params[1][pick]).cpu().numpy()
            rows.append(np.concatenate([render_mesh(v, rig.faces, cam_side, vertex_colors=np.tile(colour, (len(v), 1)),
                                                    background=(1, 1, 1))[0].numpy() for v in V], 1))
    return (np.concatenate(rows, 0) * 255).round().astype(np.uint8)


def run(glb: Path, anim: str, out: Path, tracker: str = "exact", fps: float = 24.0, seconds: float = 3.0,
        size: int = 512, points: int = 1500, device: str = "cpu", opts: FitOptions | None = None,
        yaw: float = 35.0, elevation: float = 10.0, depth: str = "none", fill: float = 0.6, follow: bool = False,
        log=print) -> dict:
    """Play ``anim`` of ``glb`` from rest, track it, fit it, and score the fit; writes ``out/bench.json`` and a
    side-on comparison, ``out/compare_side.png``. ``depth``: none, model (Depth Anything on the rendered
    video) or truth (the exact relative depth, what a perfect depth estimate would give)."""
    from PIL import Image

    opts = opts or FitOptions()
    g = AnimatedGLB(glb)
    ai = g.animations.index(anim) if anim in g.animations else int(anim)
    rig = RigModel(g, device=device)
    truth = played_from_rest(rig, g, ai, fps, seconds, follow=follow)
    with torch.no_grad():
        rest = rig.pose_points(rig.vertex_basis(), *rig.zeros(1))[0].cpu().numpy()
    cam = start_camera(rest, size, size, yaw_deg=yaw, elevation_deg=elevation, fill=fill)
    rgb, mask, face, bary = render_start(rest, rig.faces, cam, g.uvs, g.texture)
    seeds = seed_points(mask, face, bary, points, rgb=rgb)
    write_start(out, rgb, cam, seeds, mask)

    t0 = time.time()
    frames = None
    if tracker != "exact" or depth == "model":
        import imageio.v3 as iio

        frames = render_video(rig, *truth, cam, g.uvs, g.texture)
        iio.imwrite(out / "video.mp4", frames, fps=fps)
    if tracker == "exact":
        tracks, visible = track.exact(rig, *truth, seeds, cam)
    else:
        tracks, visible = track.cotracker(frames, seeds, device=device, log=log)
    t_track = time.time() - t0
    observed = None
    if depth == "truth":
        with torch.no_grad():
            _, z = project(rig.pose_points(rig.point_basis(seeds.faces, seeds.bary), *truth), cam)
        observed = (1 / z).cpu().numpy()
    elif depth == "model":
        from . import depth as depth_model

        observed = depth_model.sample(depth_model.estimate(frames, device=device, log=log), tracks)

    t0 = time.time()
    result = fit(rig, seeds, tracks, visible, cam, opts, depth=observed, log=log)
    t_fit = time.time() - t0
    fitted = (result.rotation, result.translation)
    scores = score(rig, fitted, truth, cam)
    # The objective at the truth: lower than at the fit means the optimiser stopped short of it; higher
    # means the objective itself prefers the fit, so better terms, not more iterations, are needed.
    objective = Objective(rig, seeds, tracks, visible, cam, opts, observed)
    with torch.no_grad():
        still = torch.zeros(len(tracks), 2, dtype=truth[1].dtype)
        at_truth = {k: float(v) for k, v in objective.terms(*truth, still, np.arange(len(tracks))).items()}
    side = start_camera(rest, size // 2, size // 2, yaw_deg=yaw + 90.0, elevation_deg=5.0, fill=0.75)
    Image.fromarray(contact_sheet(rig, fitted, truth, side)).save(out / "compare_side.png")
    speed = {"fitted": float(result.travel[len(result.travel) // 4:].mean(0).norm() * fps),
             "planted": contact_speed(rig, *truth, fps), "hub": travel_speed(rig, g, ai)}
    report = {"glb": str(glb), "anim": g.animations[ai], "tracker": tracker, "depth": depth, "fill": fill,
              "follow": follow,
              "frames": len(tracks), "joints": rig.n_joints, "hub": rig.names[rig.hub],
              "visible_share": float(np.mean(visible)), "speed": speed,
              "seconds": {"track": round(t_track, 1), "fit": round(t_fit, 1)}, **result.report,
              "loss_at_truth": at_truth, "error": scores}
    (out / "bench.json").write_text(json.dumps(report, indent=2))
    f, r = scores["fit"], scores["rest"]
    log(f"[bench] {Path(glb).stem}/{g.animations[ai]} ({tracker}, depth {depth}): "
        f"vertex error {f['vertex_mean']:.2%} of size (across view {f['across_view']:.2%}, "
        f"along {f['along_view']:.2%}; standing still: {r['vertex_mean']:.2%}); "
        f"travel {speed['fitted']:.3g}/s (feet slide at {speed['planted'] or 0:.3g}/s, "
        f"hub moves at {speed['hub']:.3g}/s); fit {t_fit:.0f}s; "
        f"loss at the fit {sum(result.report['loss'].values()):.4f}, at the truth {sum(at_truth.values()):.4f}")
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--glb", required=True, help="a skinned glTF with the animation")
    ap.add_argument("--anim", required=True, help="animation name or index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tracker", choices=["exact", "cotracker"], default="exact")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--points", type=int, default=1500)
    ap.add_argument("--iterations", type=int, default=FitOptions.iterations)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--depth", choices=["none", "model", "truth"], default="none",
                    help="relative depth for the fit: none, Depth Anything on the rendered video, or the truth")
    ap.add_argument("--depth-weight", type=float, default=FitOptions.depth)
    ap.add_argument("--yaw", type=float, default=35.0)
    ap.add_argument("--elevation", type=float, default=10.0)
    ap.add_argument("--fill", type=float, default=0.6, help="share of the frame the character spans")
    ap.add_argument("--follow", action="store_true",
                    help="the camera follows the character, as a video that returns to its start frame tends to")
    ap.add_argument("--economy", type=float, default=FitOptions.economy)
    ap.add_argument("--ground", type=float, default=FitOptions.ground)
    ap.add_argument("--slide", type=float, default=FitOptions.slide)
    a = ap.parse_args(argv)
    run(Path(a.glb), a.anim, Path(a.out), a.tracker, seconds=a.seconds, size=a.size, points=a.points,
        device=a.device, opts=FitOptions(iterations=a.iterations, economy=a.economy, ground=a.ground, slide=a.slide,
                                         depth=a.depth_weight), yaw=a.yaw, elevation=a.elevation, depth=a.depth,
        fill=a.fill, follow=a.follow)


if __name__ == "__main__":
    main()
