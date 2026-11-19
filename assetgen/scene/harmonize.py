"""Even out the look of separately generated props, so they read as one scene.

Each prop's picture was drawn on its own, so brightness, contrast, saturation and the
colour of the light drift from one to the next even with a shared style prompt. This
measures each prop's surface colour (sampled over its surface by area, not over its
texture atlas, whose unused space would skew it) in CIELAB and moves it part of the
way towards a reference: the scene's concept picture when there is one, otherwise
the props pooled together.

Only the lighting is matched, not the colours themselves: lightness (mean and
contrast) and saturation are matched in CIELAB, and the colour of the light is read
off the near-grey surfaces (stone, metal, cloth), which show the light's colour rather
than the object's, and balanced with per-channel gains, as a camera's white balance
does. A red barrel stays red; it just stops looking lit by a different sun.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mesh import TexturedMesh

NEUTRAL_CHROMA = 12.0     # CIELAB chroma below which a surface counts as grey
LIMITS = (0.6, 1.6)       # contrast and saturation change at most this much


@dataclass
class Look:
    lightness: float      # mean L*
    contrast: float       # spread of L*
    saturation: float     # mean chroma
    balance: np.ndarray   # mean RGB of the near-grey surfaces over its own mean: the light's colour

    def to_dict(self) -> dict:
        return {"lightness": round(self.lightness, 2), "contrast": round(self.contrast, 2),
                "saturation": round(self.saturation, 2), "balance": [round(float(v), 3) for v in self.balance]}


def surface_colors(m: TexturedMesh, n: int = 20000, seed: int = 0) -> np.ndarray:
    """RGB (0-1) of ``n`` points spread over the surface by area."""
    rng = np.random.default_rng(seed)
    v = m.vertices[m.faces].astype(np.float64)
    area = np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    if area.sum() <= 0:
        return np.zeros((0, 3))
    faces = rng.choice(len(m.faces), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w = np.stack([1 - s, s * (1 - r2), s * r2], axis=1)
    uv = (m.uvs[m.faces[faces]] * w[..., None]).sum(1)
    H, W = m.texture.shape[:2]
    px = np.clip((uv[:, 0] % 1.0) * W, 0, W - 1).astype(int)
    py = np.clip((uv[:, 1] % 1.0) * H, 0, H - 1).astype(int)
    return m.texture[py, px, :3].astype(np.float64) / 255.0


def look(rgb: np.ndarray) -> Look:
    from skimage.color import rgb2lab

    rgb = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    lab = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    grey = rgb[(chroma < NEUTRAL_CHROMA) & (lab[:, 0] > 5)]
    mean = grey.mean(0) if len(grey) >= 50 else np.ones(3)
    return Look(float(lab[:, 0].mean()), float(lab[:, 0].std()), float(chroma.mean()), mean / max(mean.mean(), 1e-6))


def image_look(image: np.ndarray) -> Look:
    return look(np.asarray(image, dtype=np.float64)[..., :3].reshape(-1, 3) / 255.0)


def pooled(looks: list[Look]) -> Look:
    """The middle of several looks (medians, so one odd prop does not pull the rest)."""
    return Look(float(np.median([x.lightness for x in looks])), float(np.median([x.contrast for x in looks])),
                float(np.median([x.saturation for x in looks])), np.median([x.balance for x in looks], axis=0))


def apply(texture: np.ndarray, own: Look, target: Look, strength: float) -> np.ndarray:
    """``texture`` (H, W, 3 uint8) moved ``strength`` (0-1) of the way from ``own`` look to ``target``."""
    from skimage.color import lab2rgb, rgb2lab

    if strength <= 0:
        return texture
    t = float(np.clip(strength, 0, 1))
    gains = np.clip(target.balance / np.maximum(own.balance, 1e-6), *LIMITS) ** t
    lab = rgb2lab(np.clip(texture[..., :3].astype(np.float64) / 255.0 * gains, 0, 1))
    contrast = np.clip(target.contrast / max(own.contrast, 1e-6), *LIMITS) ** t
    saturation = np.clip(target.saturation / max(own.saturation, 1e-6), *LIMITS) ** t
    L = (lab[..., 0] - own.lightness) * contrast + own.lightness + t * (target.lightness - own.lightness)
    out = np.concatenate([np.clip(L, 0, 100)[..., None], lab[..., 1:] * saturation], axis=-1)
    return (np.clip(lab2rgb(out), 0, 1) * 255 + 0.5).astype(np.uint8)
