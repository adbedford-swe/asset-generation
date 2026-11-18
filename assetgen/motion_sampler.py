"""Fixed-step flow sampling for UniMate, with seamless loops enforced inside the sampler.

UniMate's flow is linear, ``x_t = (1 - t) eps + t x1``, and the network predicts the
velocity ``v = x1 - eps``. Every Euler step can therefore be written as "predict the
clean motion, then move to the next noise level":

    x1_hat  = x_t + (1 - t) v          eps_hat = x_t - t v
    x_next  = (1 - t_next) eps_hat + t_next x1_hat        (identical to x_t + dt v)

A loop of L frames is enforced by making ``x1_hat`` periodic before each move: frames
t, t + L, t + 2L, ... of the 60-frame window are replaced by their mean. Every
per-frame feature UniMate uses is invariant to world position (root-relative joint
positions, local rotations, facing-frame velocities, root height), so equal features
mean the same pose moving the same way, and the root still travels for a walk. The
model shapes the whole motion around the constraint at every step, so the result is
not a blend: at t = 1 the window is exactly periodic and frames 0..L play as a loop
whose seam matches in position, velocity and acceleration.

L is chosen from the motion itself. A first pass samples freely; its dominant period
(autocorrelation of the joints' features) gives the cycle length, and L is the largest
whole number of cycles that leaves a margin of overlap. Motions with no clear cycle
(idle, attack) loop over the window minus that margin. The first pass runs at a quarter
of the steps; the second reuses its noise, so it stays as close as possible to the
motion the prompt gives, and a batch with loops costs about 1.25x a plain one.

Also replaces UniMate's default adaptive dopri5 solve (atol 1e-6) for plain samples with
the same fixed-step Euler integration UniMate already uses for in-betweening.

Configured through the environment by the animate stage:
    ASSETGEN_FLOW_STEPS   Euler steps (default 32; 0 keeps UniMate's own dopri5 solve). Against
                          dopri5 on the same noise, 50 steps differ by 0.3-1.2% and 25 by 0.5-2.4%
                          of the features' spread, at a third to a seventh of the time.
    ASSETGEN_LOOP         JSON {"prompts": [...], "frames": "auto" | int, "min_overlap": int}
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

DEFAULT_STEPS = 32
MIN_OVERLAP = 12
MIN_PERIOD = 12


def _interval(transport):
    return transport.check_interval(transport.train_eps, transport.sample_eps, sde=False, eval=True,
                                    reverse=False, last_step_size=0.0)


def periodic(x1: torch.Tensor, periods: list[int | None]) -> torch.Tensor:
    """Average frames t, t+L, t+2L, ... of each sample with a period (time is the last axis)."""
    T = x1.shape[-1]
    out = x1.clone()
    for b, L in enumerate(periods):
        if not L:
            continue
        group = torch.arange(T, device=x1.device) % L
        onehot = torch.nn.functional.one_hot(group, L).to(x1.dtype)          # (T, L)
        mean = (x1[b] @ onehot) / onehot.sum(0)                               # (J, D, L)
        out[b] = mean[..., group]
    return out


def flow_sample(sample_model, transport, cond, eps: torch.Tensor, steps: int,
                periods: list[int | None] | None = None) -> torch.Tensor:
    t0, t1 = _interval(transport)
    ts = torch.linspace(t0, t1, steps + 1, device=eps.device)
    x = eps.clone()
    loop = periods is not None and any(periods)
    for i in range(steps):
        t, tn = float(ts[i]), float(ts[i + 1])
        v = sample_model(x, torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype), cond=cond)
        if loop:
            x1 = periodic(x + (1 - t) * v, periods)
            x = (1 - tn) * (x - t * v) + tn * x1
        else:
            x = x + (tn - t) * v
    return x


def dominant_period(signal: np.ndarray, min_period: int = MIN_PERIOD, max_period: int | None = None,
                    threshold: float = 0.5) -> int | None:
    """Period (frames) of a (T, C) signal from its normalised autocorrelation, or None if it has no clear cycle."""
    T = signal.shape[0]
    s = signal - signal.mean(0, keepdims=True)
    max_period = max_period or T - 8
    best, best_r = None, threshold
    for lag in range(min_period, max_period + 1):
        a, b = s[:-lag], s[lag:]
        r = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
        if r > best_r:
            best, best_r = lag, r
    return best


def loop_length(signal: np.ndarray, T: int, frames, min_overlap: int = MIN_OVERLAP) -> int:
    """Loop length for one sample: as many whole cycles as fit with ``min_overlap`` frames to spare."""
    if isinstance(frames, int) and frames > 0:
        return max(MIN_PERIOD, min(frames, T - min_overlap))
    room = T - min_overlap
    # Lags past half the window overlap too few frames to measure; a longer natural cycle
    # becomes one cycle over the loop, and the model sets the cadence to fit.
    period = dominant_period(signal, max_period=min(room, T // 2))
    if period is None:
        return room
    return period * max(1, room // period)


def generate_samples(model, cond, motion_shape, diff_model, diffusion, gen_diffusion=None, device=None,
                     cfg_scale=1.0, x1_known=None, keep_mask=None, _original=None):
    """Drop-in for ``unimate.inference.generate.generate_samples`` (see module docstring)."""
    steps = int(os.environ.get("ASSETGEN_FLOW_STEPS", DEFAULT_STEPS))
    if x1_known is not None or keep_mask is not None or diff_model != "flow" or steps <= 0:
        return _original(model, cond, motion_shape, diff_model, diffusion, gen_diffusion=gen_diffusion,
                         device=device, cfg_scale=cfg_scale, x1_known=x1_known, keep_mask=keep_mask)
    from unimate.inference.generate import ClassifierFreeSampleModel

    device = device or next(model.parameters()).device
    sample_model = ClassifierFreeSampleModel(model, cfg_scale=cfg_scale) if cfg_scale > 1.0 else model
    spec = json.loads(os.environ.get("ASSETGEN_LOOP", "null") or "null")
    eps = torch.randn(motion_shape, device=device)
    captions = cond.get("caption") or [None] * motion_shape[0]
    prompts = (spec or {}).get("prompts") or []
    wanted = [prompts == "all" or c in prompts for c in captions]
    with torch.no_grad():
        if not any(wanted):
            return flow_sample(sample_model, gen_diffusion.transport, cond, eps, steps)
        # A quick first pass only has to reveal each motion's cycle; the second pass, from the same
        # noise and at full steps, produces every sample (one-shots come out exactly as in one pass).
        first = flow_sample(sample_model, gen_diffusion.transport, cond, eps, max(8, steps // 4))
        T = motion_shape[-1]
        n_joints = cond["n_joints"].tolist()
        periods = []
        for b, want in enumerate(wanted):
            if not want:
                periods.append(None)
                continue
            sig = first[b, :int(n_joints[b])].reshape(-1, T).T.float().cpu().numpy()
            periods.append(loop_length(sig, T, spec.get("frames", "auto"), spec.get("min_overlap", MIN_OVERLAP)))
        print(f"[loop] periods per sample: {periods}", flush=True)
        return flow_sample(sample_model, gen_diffusion.transport, cond, eps, steps, periods)


def install() -> None:
    """Route UniMate's sampler through ``generate_samples`` above (before sample.py is imported)."""
    import unimate.inference.generate as gen

    original = gen.generate_samples

    def patched(*args, **kwargs):
        return generate_samples(*args, _original=original, **kwargs)

    gen.generate_samples = patched


def loop_period(motion: np.ndarray, rel_tol: float = 1e-4) -> int | None:
    """Exact period of a sampled motion (T, J, D) whose frames t and t+L were tied, or None."""
    T = motion.shape[0]
    scale = float(np.abs(motion).max()) or 1.0
    for L in range(MIN_PERIOD, T):
        if np.abs(motion[:T - L] - motion[L:]).max() <= rel_tol * scale:
            return L
    return None
