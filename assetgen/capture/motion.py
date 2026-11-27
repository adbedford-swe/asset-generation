"""What a fitted clip does, measured: where it touches the ground, how fast it travels, where it loops,
and when its action happens.

None of this needs to know what the character is or what the clip shows. Contacts are surface points
resting on the ground; footsteps are contacts beginning, attributed to the joint that carries them.
A loop is the stretch between two moments of the clip that look alike. A one-shot's phases come from
how fast its fastest parts move: winding up, striking, recovering. The soundtrack a video model makes
with the picture marks impacts too, and is kept as the clip's sound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Motion:
    rotation: torch.Tensor     # (T, J, 3)
    translation: torch.Tensor  # (T, 3) hub
    travel: torch.Tensor       # (T, 2) world x, z velocity per frame

    def __len__(self) -> int:
        return len(self.rotation)

    def slice(self, a: int, b: int) -> "Motion":
        return Motion(self.rotation[a:b], self.translation[a:b], self.travel[a:b])

    def offsets(self) -> np.ndarray:
        """Accumulated travel (T, 3) in world units, from frame 0."""
        v = self.travel.detach().cpu().double().numpy()
        out = np.zeros((len(v), 3))
        out[1:, 0] = np.cumsum(v[:-1, 0])
        out[1:, 2] = np.cumsum(v[:-1, 1])
        return out


def posed_world(rig, m: Motion, basis: torch.Tensor) -> np.ndarray:
    """World positions (T, N, 3) of the points behind ``basis``, travel included."""
    with torch.no_grad():
        p = rig.pose_points(basis, m.rotation, m.translation).cpu().double().numpy()
    return p + m.offsets()[:, None]


def contacts(rig, m: Motion, fps: float, band: float = 0.015, still: float = 0.25) -> list[dict]:
    """Ground contacts: intervals where some of the surface rests on the ground (within ``band`` of the
    asset's size) and keeps still (moving less than ``still`` asset sizes a second), grouped by the joint
    that carries those points. ``[{"joint", "start", "end"}]`` in seconds."""
    B = rig.vertex_basis()
    P = posed_world(rig, m, B)
    floor = float(posed_world(rig, m.slice(0, 1), B)[0, :, 1].min())
    height = P[..., 1] - floor
    speed = np.zeros(height.shape)
    speed[1:] = np.linalg.norm(np.diff(P[..., [0, 2]], axis=0), axis=-1) * fps
    speed[0] = speed[1] if len(speed) > 1 else 0
    resting = (height < band * rig.size) & (speed < still * rig.size)
    owner = rig.vertex_joints[np.arange(len(rig.vertex_joints)), rig.vertex_weights.argmax(1)]
    out = []
    for j in np.unique(owner[resting.any(0)]):
        frames = resting[:, owner == j].sum(1) >= max(3, int(0.02 * (owner == j).sum()))
        for start, end in _runs(frames, min_length=2):
            out.append({"joint": rig.names[j], "start": round(start / fps, 4), "end": round(end / fps, 4)})
    return sorted(out, key=lambda c: c["start"])


def _runs(flags: np.ndarray, min_length: int = 1) -> list[tuple[int, int]]:
    """(start, end) frame pairs of runs of True, end inclusive."""
    runs, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_length:
                runs.append((start, i - 1))
            start = None
    return runs


def hub_parent_rotation(rig) -> torch.Tensor:
    """World rotation (3, 3) of the frame the hub's translation is in, at rest."""
    p = rig.parent[rig.hub]
    if p < 0:
        return rig.base[rig.hub][:3, :3]
    return rig.world(*rig.zeros(1))[0, p, :3, :3]


def _features(rig, m: Motion) -> np.ndarray:
    """Per frame: every joint's rotation and the hub's height. Ground covered is left out, so a walk's
    second step matches its first."""
    r = m.rotation.detach().cpu().double().numpy().reshape(len(m), -1)
    with torch.no_grad():
        up = hub_parent_rotation(rig).double().cpu().numpy()[1]
    height = m.translation.detach().cpu().double().numpy() @ up / rig.size
    return np.concatenate([r, 3 * height[:, None]], 1)


def loop(rig, m: Motion, fps: float, skip: float = 0.5, skip_end: float = 0.0, min_seconds: float = 0.4,
         max_seconds: float | None = None) -> tuple[Motion, dict]:
    """The best single cycle of a clip, closed into a loop of L + 1 frames whose last equals its first.

    The first ``skip`` seconds (the ease out of the start frame's rest pose) and the last ``skip_end``
    (an ease back into it) are left out. The period is the lag at which the motion best matches itself;
    the cycle starts where its two ends match best, and what difference remains is spread over it."""
    f = _features(rig, m)
    T = len(f) - min(int(round(skip_end * fps)), len(f) // 3)
    f = f[:T]
    s0 = min(int(round(skip * fps)), T // 3)
    lo = max(2, int(round(min_seconds * fps)))
    hi = min(T - s0 - 1, int(round(max_seconds * fps)) if max_seconds else T)
    if hi <= lo:
        raise ValueError(f"a {T}-frame clip is too short to find a loop of at least {lo} frames in")
    spread = float(np.mean(np.linalg.norm(f[s0:] - f[s0:].mean(0), axis=1))) or 1.0
    best = None
    for L in range(lo, hi + 1):
        a = f[s0:T - L]
        b = f[s0 + L:T]
        d = np.linalg.norm(a - b, axis=1)
        # Prefer the lag whose whole overlap matches, then the start where the seam matches best.
        score = float(d.mean()) / spread + 0.002 * L / fps
        if best is None or score < best[0]:
            best = (score, L, s0 + int(d.argmin()))
    _, L, s = best
    cut = m.slice(s, s + L + 1)
    ramp = torch.linspace(0, 1, L + 1, dtype=cut.rotation.dtype, device=cut.rotation.device)
    rot = cut.rotation - ramp[:, None, None] * (cut.rotation[-1] - cut.rotation[0])[None]
    shift = cut.translation[-1] - cut.translation[0]
    tr = cut.translation - ramp[:, None] * shift[None]
    # Over a cycle the hub comes back to where it was, except for ground covered: that is travel, kept as
    # a steady velocity along with the travel the fit already found (a camera that followed the character).
    with torch.no_grad():
        parent = hub_parent_rotation(rig).to(shift)
    covered = (parent @ shift)[[0, 2]] / L
    trav = (cut.travel[:-1].mean(0) + covered).expand(L + 1, 2).clone()
    seam = float(np.linalg.norm(f[s] - f[s + L])) / spread
    return Motion(rot, tr, trav), {"start": s, "frames": L, "seconds": round(L / fps, 4), "seam": round(seam, 4)}


def speed_profile(rig, m: Motion, fps: float, share: float = 0.05) -> np.ndarray:
    """Per frame, the mean speed (asset sizes per second) of the fastest ``share`` of the surface,
    travel left out: how fast the clip's most active parts move."""
    B = rig.vertex_basis()
    with torch.no_grad():
        P = rig.pose_points(B, m.rotation, m.translation).cpu().double().numpy()
    v = np.linalg.norm(np.diff(P, axis=0), axis=-1) * fps / rig.size          # (T-1, V)
    k = max(1, int(share * v.shape[1]))
    top = -np.sort(-v, axis=1)[:, :k].mean(1)
    return np.r_[top[:1], top]


def phases(rig, m: Motion, fps: float, onsets: list[float] | None = None) -> dict:
    """A one-shot's timing: ``windup`` (start to the action), ``active`` (while its fastest parts move at
    more than half their peak speed), ``recovery`` (after), and ``impact``, the moment the action lands:
    the strongest sound onset inside the active phase when there is a soundtrack, otherwise the last fast
    frame before the sharpest slowing-down after the peak. ``onsets`` are strongest first, as
    :func:`onsets` gives them. Times in seconds."""
    sp = speed_profile(rig, m, fps)
    peak = int(sp.argmax())
    over = sp > 0.5 * sp[peak]
    a = peak
    while a > 0 and over[a - 1]:
        a -= 1
    b = peak
    while b < len(sp) - 1 and over[b + 1]:
        b += 1
    end = (len(sp) - 1) / fps
    impact = None
    if onsets:
        inside = [o for o in onsets if a / fps - 0.1 <= o <= b / fps + 0.1]
        impact = inside[0] if inside else None
    if impact is None:
        # The last fast frame before the sharpest slowing-down: where the swing is stopped.
        decel = -np.diff(sp[peak:b + 2]) if b + 1 < len(sp) else -np.diff(sp[peak:])
        impact = (peak + int(decel.argmax())) / fps if len(decel) else peak / fps
    return {"windup": [0.0, round(a / fps, 4)], "active": [round(a / fps, 4), round(b / fps, 4)],
            "recovery": [round(b / fps, 4), round(end, 4)], "impact": round(float(impact), 4),
            "peak_speed": round(float(sp[peak]), 4)}


def onsets(samples: np.ndarray, rate: int, frame: float = 0.02, threshold: float = 2.5) -> list[float]:
    """Sound onsets (seconds), strongest first: peaks of spectral flux standing well above its median."""
    if samples.ndim > 1:
        samples = samples.mean(1)
    hop = max(1, int(frame * rate))
    n = 2 * hop
    if len(samples) < 2 * n:
        return []
    window = np.hanning(n)
    frames = np.lib.stride_tricks.sliding_window_view(samples, n)[::hop] * window
    mag = np.abs(np.fft.rfft(frames, axis=1))
    flux = np.r_[0, np.maximum(0, np.diff(np.log1p(mag), axis=0)).sum(1)]
    base = np.median(flux) + 1e-9
    peaks = [i for i in range(1, len(flux) - 1)
             if flux[i] > threshold * base and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]]
    peaks.sort(key=lambda i: -flux[i])
    # Window i spans [i, i + 2) hops; the first window a sound reaches holds it in its second hop.
    return [round((i + 1) * hop / rate, 4) for i in peaks]


def footsteps(found: list[dict]) -> list[dict]:
    """Events for contacts beginning: ``{"time", "type": "contact", "joint"}``."""
    return [{"time": c["start"], "type": "contact", "joint": c["joint"]} for c in found]
