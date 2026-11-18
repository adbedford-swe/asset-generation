"""Fixed-step flow sampling for UniMate, with seamless loops of any length.

UniMate denoises one 60-frame window at a time with a linear flow,
``x_t = (1 - t) eps + t x1``, the network predicting the velocity ``v = x1 - eps``.
Here every motion is a *strip* of frames covered by 60-frame windows. Each Euler
step runs all windows of all motions through the model as one batch, blends each
frame's velocity over the windows that cover it (weighted towards window centres),
and moves the whole strip:

* a one-shot clip is a plain strip of one window: ordinary sampling;
* a loop of L frames is a *ring*: windows wrap around its end, so frame L follows
  frame L - 1 exactly as frame 1 follows frame 0. The motion is periodic by
  construction, with no seam to blend, and windows overlap so neighbouring stretches
  stay coherent; their positions shift every step so no window edge settles into the
  motion (the MultiDiffusion idea, on a circular time axis). A ring shorter than a
  window simply wraps within it, which makes the window exactly periodic.

Every per-frame feature UniMate uses is independent of world position (root-relative
joint positions, local rotations, facing-frame velocities, root height), so frames
that agree are the same pose moving the same way while the root still travels. The
result loops with the seam matched in position, velocity and acceleration.

The loop length comes from the motion. A quick free pass over a longer strip (three
overlapping windows, 120 frames) reveals the motion's cycle up to 60 frames long
(autocorrelation of its joint features); the loop is one cycle, or the fewest whole
cycles that last at least ``MIN_LOOP`` frames. A motion without a cycle (idle, an
attack, or a model that does not settle into a gait) loops over ``DEFAULT_LOOP``
frames. An explicit length can be any number of frames up to ``MAX_LOOP``.

This also replaces UniMate's default adaptive dopri5 solve (atol 1e-6) with fixed-step
Euler, the integration UniMate itself uses for in-betweening.

Configured through the environment by the animate stage:
    ASSETGEN_FLOW_STEPS   Euler steps (default 32; 0 keeps UniMate's own dopri5 solve). Against
                          dopri5 on the same noise, 50 steps differ by 0.3-1.2% and 25 by 0.5-2.4%
                          of the features' spread, at a third to a seventh of the time.
    ASSETGEN_LOOP         JSON {"prompts": [...] | "all", "frames": "auto" | int}
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

DEFAULT_STEPS = 32
WINDOW = 60
DEFAULT_LOOP = 90     # loop length (3 s) for motions without a cycle
MIN_LOOP = 20         # shortest loop built from whole cycles
MIN_PERIOD = 12
MAX_LOOP = 600        # 20 s at 30 fps
PROBE = 120           # strip length of the free pass that finds a motion's cycle


@dataclass
class Strip:
    """One motion being sampled: ``frames`` long, a ring if ``circular``, covered by windows at ``starts``."""
    frames: int
    circular: bool
    starts: list[int] = field(default_factory=list)

    def __post_init__(self):
        if self.starts:
            return
        if self.circular:
            n = max(1, math.ceil(self.frames / (WINDOW / 2)))
            self.starts = [round(i * self.frames / n) for i in range(n)]
        else:
            n = max(1, math.ceil((self.frames - WINDOW) / (WINDOW / 2)) + 1)
            last = max(0, self.frames - WINDOW)
            self.starts = [round(i * last / max(1, n - 1)) for i in range(n)]


def _interval(transport):
    return transport.check_interval(transport.train_eps, transport.sample_eps, sde=False, eval=True,
                                    reverse=False, last_step_size=0.0)


def _select(cond: dict, index: list[int], batch: int) -> dict:
    """Rows ``index`` of every per-sample entry of a UniMate conditioning dict."""
    out = {}
    for k, v in cond.items():
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == batch:
            out[k] = v.index_select(0, torch.tensor(index, device=v.device))
        elif isinstance(v, (list, tuple)) and len(v) == batch:
            out[k] = type(v)(v[i] for i in index)
        else:
            out[k] = v
    return out


def window_weights(device) -> torch.Tensor:
    j = torch.arange(WINDOW, device=device, dtype=torch.float32)
    return 0.05 + torch.sin(math.pi * (j + 0.5) / WINDOW) ** 2


def strip_sample(sample_model, transport, cond: dict, strips: list[Strip], noise: list[torch.Tensor], steps: int,
                 generator: torch.Generator | None = None) -> list[torch.Tensor]:
    """Euler-integrate every strip; ``noise[b]`` is (J, D, frames). Returns the clean strips."""
    batch = len(strips)
    device = noise[0].device
    t0, t1 = _interval(transport)
    ts = torch.linspace(float(t0), float(t1), steps + 1)
    xs = [n.clone() for n in noise]
    owner = [b for b, s in enumerate(strips) for _ in s.starts]
    wcond = _select(cond, owner, batch)
    weights = window_weights(device)
    offsets = torch.arange(WINDOW, device=device)
    for i in range(steps):
        t, tn = float(ts[i]), float(ts[i + 1])
        index = []
        for b, s in enumerate(strips):
            shift = 0
            if s.circular and len(s.starts) > 1 and generator is not None:
                shift = int(torch.randint(0, s.frames, (1,), generator=generator))
            for start in s.starts:
                pos = start + shift + offsets
                index.append(pos % s.frames if s.circular else pos.clamp_max(s.frames - 1))
        windows = torch.stack([xs[b][..., idx] for b, idx in zip(owner, index)])
        tb = torch.full((len(owner),), t, device=device, dtype=windows.dtype)
        v = sample_model(windows, tb, cond=wcond)
        acc = [torch.zeros_like(x) for x in xs]
        wsum = [torch.zeros(x.shape[-1], device=device) for x in xs]
        for w, (b, idx) in enumerate(zip(owner, index)):
            acc[b].index_add_(-1, idx, v[w] * weights)
            wsum[b].index_add_(0, idx, weights)
        for b in range(batch):
            xs[b] = xs[b] + (tn - t) * acc[b] / wsum[b]
    return xs


def dominant_period(signal: np.ndarray, min_period: int = MIN_PERIOD, max_period: int | None = None,
                    threshold: float = 0.5) -> int | None:
    """Period (frames) of a (T, C) signal from its normalised autocorrelation, or None if it has no clear cycle.

    Multiples of a cycle correlate as well as the cycle itself, so this is the first local peak that comes
    close to the strongest one: the fundamental, not a multiple of it."""
    T = signal.shape[0]
    s = signal - signal.mean(0, keepdims=True)
    spread = s.std(0)
    s = s[:, spread > 1e-6] / spread[spread > 1e-6]  # small, periodic channels (legs) count as much as large ones
    if s.shape[1] == 0:
        return None
    max_period = max_period or T // 2
    lags = list(range(min_period - 1, max_period + 2))
    r = []
    for lag in lags:
        a, b = s[:-lag], s[lag:]
        r.append(float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12)))
    peaks = [(lags[i], r[i]) for i in range(1, len(lags) - 1)
             if r[i] >= r[i - 1] and r[i] >= r[i + 1] and r[i] > threshold]
    if not peaks:
        return None
    strongest = max(v for _, v in peaks)
    return next(lag for lag, v in peaks if v >= 0.9 * strongest)


def loop_length(signal: np.ndarray | None, frames) -> int:
    """Loop length for one motion: explicit, else from its cycle (see the module docstring)."""
    if isinstance(frames, int) and frames > 0:
        return max(MIN_PERIOD, min(frames, MAX_LOOP))
    period = dominant_period(signal, max_period=signal.shape[0] // 2) if signal is not None else None
    if period is None:
        return DEFAULT_LOOP
    return period * math.ceil(MIN_LOOP / period)


def _tile(x: torch.Tensor, frames: int) -> torch.Tensor:
    return x[..., torch.arange(frames, device=x.device) % x.shape[-1]]


def generate_samples(model, cond, motion_shape, diff_model, diffusion, gen_diffusion=None, device=None,
                     cfg_scale=1.0, x1_known=None, keep_mask=None, _original=None):
    """Drop-in for ``unimate.inference.generate.generate_samples`` (see module docstring).

    Returns (B, J, D, T) with T >= 60: one-shots in their first 60 frames (the rest holds the last
    pose), loops as their ring repeated, so the animate stage finds each loop's exact period."""
    steps = int(os.environ.get("ASSETGEN_FLOW_STEPS", DEFAULT_STEPS))
    if x1_known is not None or keep_mask is not None or diff_model != "flow" or steps <= 0:
        return _original(model, cond, motion_shape, diff_model, diffusion, gen_diffusion=gen_diffusion,
                         device=device, cfg_scale=cfg_scale, x1_known=x1_known, keep_mask=keep_mask)
    from unimate.inference.generate import ClassifierFreeSampleModel

    device = device or next(model.parameters()).device
    sample_model = ClassifierFreeSampleModel(model, cfg_scale=cfg_scale) if cfg_scale > 1.0 else model
    spec = json.loads(os.environ.get("ASSETGEN_LOOP", "null") or "null") or {}
    B, J, D, T = motion_shape
    captions = cond.get("caption") or [None] * B
    prompts = spec.get("prompts") or []
    wanted = [prompts == "all" or c in prompts for c in captions]
    transport = gen_diffusion.transport
    generator = torch.Generator().manual_seed(torch.initial_seed() % (2 ** 63))
    eps = torch.randn(motion_shape, device=device)

    with torch.no_grad():
        if not any(wanted):
            strips = [Strip(T, False) for _ in range(B)]
            return torch.stack(strip_sample(sample_model, transport, cond, strips, list(eps), steps))

        frames = spec.get("frames", "auto")
        if frames == "auto":
            # A quick free pass over a longer strip reveals each motion's cycle.
            probe = [Strip(PROBE, False) for _ in range(B)]
            probe_noise = [torch.randn(J, D, PROBE, device=device) for _ in range(B)]
            free = strip_sample(sample_model, transport, cond, probe, probe_noise, max(8, steps // 4))
            n_joints = cond["n_joints"].tolist()
            lengths = []
            for b in range(B):
                sig = free[b][:int(n_joints[b])].reshape(-1, PROBE).T.float().cpu().numpy()
                lengths.append(loop_length(sig, "auto") if wanted[b] else None)
        else:
            lengths = [loop_length(None, int(frames)) if w else None for w in wanted]
        print(f"[loop] loop lengths per sample: {lengths}", flush=True)

        strips, noise = [], []
        for b in range(B):
            if lengths[b]:
                strips.append(Strip(lengths[b], True))
                noise.append(_tile(eps[b], lengths[b]) if lengths[b] <= T else
                             torch.randn(J, D, lengths[b], device=device))
            else:
                strips.append(Strip(T, False))
                noise.append(eps[b])
        out = strip_sample(sample_model, transport, cond, strips, noise, steps, generator)

    longest = max([T] + [L + MIN_PERIOD for L in lengths if L])
    result = []
    for b, x in enumerate(out):
        if lengths[b]:
            result.append(_tile(x, longest))
        else:
            result.append(torch.cat([x, x[..., -1:].expand(*x.shape[:-1], longest - T)], -1))
    return torch.stack(result)


def install() -> None:
    """Route UniMate's sampler through ``generate_samples`` above (before sample.py is imported)."""
    import unimate.inference.generate as gen

    original = gen.generate_samples

    def patched(*args, **kwargs):
        return generate_samples(*args, _original=original, **kwargs)

    gen.generate_samples = patched


def loop_period(motion: np.ndarray, rel_tol: float = 1e-4) -> int | None:
    """Exact period of a sampled motion (T, J, D) that repeats a ring, or None."""
    T = motion.shape[0]
    scale = float(np.abs(motion).max()) or 1.0
    for L in range(MIN_PERIOD, T):
        if np.abs(motion[:T - L] - motion[L:]).max() <= rel_tol * scale:
            return L
    return None
