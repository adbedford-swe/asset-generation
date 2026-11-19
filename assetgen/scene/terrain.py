"""The ground a scene stands on: a heightfield, its mesh, and a texture that tiles.

The heightfield is seeded value noise summed over a few octaves (gentle hills of a
chosen size and height), flattened into pads where objects are placed by hand so a
tent or a well sits level. Ground textures from an image generator rarely tile, so
:func:`tileable` blends each picture with itself shifted by half, which hides the
seams wherever the texture repeats.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Heightfield:
    heights: np.ndarray   # (rows along z, cols along x), metres
    size: tuple           # (width along x, depth along z), metres; centred on the origin

    @property
    def spacing(self) -> tuple[float, float]:
        rows, cols = self.heights.shape
        return self.size[0] / (cols - 1), self.size[1] / (rows - 1)

    def _grid(self, x, z):
        rows, cols = self.heights.shape
        gx = (np.asarray(x, dtype=np.float64) + self.size[0] / 2) / self.size[0] * (cols - 1)
        gz = (np.asarray(z, dtype=np.float64) + self.size[1] / 2) / self.size[1] * (rows - 1)
        return np.clip(gx, 0, cols - 1), np.clip(gz, 0, rows - 1)

    def height(self, x, z):
        """Bilinear height at (x, z); outside the ground, the height of its nearest edge."""
        gx, gz = self._grid(x, z)
        rows, cols = self.heights.shape
        x0 = np.minimum(np.floor(gx).astype(int), cols - 2)
        z0 = np.minimum(np.floor(gz).astype(int), rows - 2)
        fx, fz = gx - x0, gz - z0
        h = self.heights
        return ((h[z0, x0] * (1 - fx) + h[z0, x0 + 1] * fx) * (1 - fz)
                + (h[z0 + 1, x0] * (1 - fx) + h[z0 + 1, x0 + 1] * fx) * fz)

    def normal(self, x, z) -> np.ndarray:
        """The unit surface normal at (x, z)."""
        dx, dz = self.spacing
        sx = (self.height(x + dx, z) - self.height(x - dx, z)) / (2 * dx)
        sz = (self.height(x, z + dz) - self.height(x, z - dz)) / (2 * dz)
        n = np.array([-sx, 1.0, -sz], dtype=np.float64)
        return n / np.linalg.norm(n)


def value_noise(shape: tuple[int, int], cells: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """Smoothly interpolated random values on a lattice of ``cells`` (x, z) across the grid, in [-1, 1]."""
    rows, cols = shape
    lattice = rng.uniform(-1, 1, size=(int(np.ceil(cells[1])) + 2, int(np.ceil(cells[0])) + 2))
    gx = np.linspace(0, cells[0], cols)
    gz = np.linspace(0, cells[1], rows)
    x0, z0 = np.floor(gx).astype(int), np.floor(gz).astype(int)
    fx, fz = gx - x0, gz - z0
    sx, sz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)  # smoothstep, so slopes are continuous
    a = lattice[np.ix_(z0, x0)]
    b = lattice[np.ix_(z0, x0 + 1)]
    c = lattice[np.ix_(z0 + 1, x0)]
    d = lattice[np.ix_(z0 + 1, x0 + 1)]
    top = a + (b - a) * sx[None, :]
    bottom = c + (d - c) * sx[None, :]
    return top + (bottom - top) * sz[:, None]


def heightfield(size, relief: float = 0.0, feature: float = 8.0, resolution: float = 0.5,
                seed: int = 0, octaves: int = 4) -> Heightfield:
    """Rolling ground ``size`` = (width, depth) metres: hills about ``feature`` metres across, ``relief``
    metres from the lowest to the highest point, centred on height 0."""
    width, depth = float(size[0]), float(size[1])
    cols = max(2, int(round(width / resolution)) + 1)
    rows = max(2, int(round(depth / resolution)) + 1)
    h = np.zeros((rows, cols))
    if relief > 0:
        rng = np.random.default_rng(seed)
        amplitude, scale = 1.0, 1.0
        for _ in range(octaves):
            h += amplitude * value_noise((rows, cols), (width / feature * scale, depth / feature * scale), rng)
            amplitude *= 0.45
            scale *= 2.1
        h -= h.min()
        h *= relief / max(h.max(), 1e-9)
        h -= relief / 2
    return Heightfield(h, (width, depth))


def flatten(field: Heightfield, pads: list[tuple[float, float, float]], falloff: float = 1.5) -> Heightfield:
    """Level the ground under each pad (x, z, radius) to the average height there, easing back into the
    hills over ``falloff`` times the radius."""
    if not pads:
        return field
    rows, cols = field.heights.shape
    xs = np.linspace(-field.size[0] / 2, field.size[0] / 2, cols)
    zs = np.linspace(-field.size[1] / 2, field.size[1] / 2, rows)
    X, Z = np.meshgrid(xs, zs)
    h = field.heights.copy()
    for x, z, radius in pads:
        if radius <= 0:
            continue
        d = np.hypot(X - x, Z - z)
        inside = d <= radius
        level = float(h[inside].mean()) if inside.any() else float(field.height(x, z))
        t = np.clip((d - radius) / max(radius * falloff, 1e-6), 0, 1)
        w = 1 - t * t * (3 - 2 * t)
        h = h * (1 - w) + level * w
    return Heightfield(h, field.size)


def ground_mesh(field: Heightfield, tile: float):
    """Vertices (V, 3), faces (F, 3), normals (V, 3) and UVs (V, 2) of the ground. UVs are in metres / ``tile``,
    so the texture repeats every ``tile`` metres (glTF's v runs down the image, here along +z)."""
    rows, cols = field.heights.shape
    xs = np.linspace(-field.size[0] / 2, field.size[0] / 2, cols)
    zs = np.linspace(-field.size[1] / 2, field.size[1] / 2, rows)
    X, Z = np.meshgrid(xs, zs)
    vertices = np.stack([X, field.heights, Z], axis=-1).reshape(-1, 3)
    idx = np.arange(rows * cols).reshape(rows, cols)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    # Counter-clockwise seen from above (+Y), so the faces point up.
    faces = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    gz, gx = np.gradient(field.heights, zs, xs)
    normals = np.stack([-gx, np.ones_like(gx), -gz], axis=-1).reshape(-1, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    uvs = np.stack([X / tile, Z / tile], axis=-1).reshape(-1, 2)
    return vertices.astype(np.float32), faces.astype(np.uint32), normals.astype(np.float32), uvs.astype(np.float32)


def tileable(image: np.ndarray, border: float = 0.25) -> np.ndarray:
    """``image`` (H, W, 3) made to repeat without visible seams: blended with itself shifted by half, the
    shifted copy (continuous across the wrap) taking over towards the edges."""
    img = np.asarray(image, dtype=np.float64)
    H, W = img.shape[:2]
    shifted = np.roll(img, (H // 2, W // 2), axis=(0, 1))

    def ramp(n):
        t = np.minimum(np.arange(n) + 0.5, n - np.arange(n) - 0.5) / (n * border)
        t = np.clip(t, 0, 1)
        return t * t * (3 - 2 * t)

    weight = np.minimum.outer(ramp(H), ramp(W))[..., None]
    return np.clip(img * weight + shifted * (1 - weight), 0, 255).astype(np.uint8)


def seam_error(image: np.ndarray) -> float:
    """How visible the wrap is: the mean jump across the right/left and bottom/top edges, against the mean
    jump between neighbouring pixels inside the image (1 is as smooth as the inside)."""
    img = np.asarray(image, dtype=np.float64)
    inside = (np.abs(np.diff(img, axis=1)).mean() + np.abs(np.diff(img, axis=0)).mean()) / 2
    wrap = (np.abs(img[:, 0] - img[:, -1]).mean() + np.abs(img[0] - img[-1]).mean()) / 2
    return float(wrap / max(inside, 1e-9))
