import numpy as np
import pytest
import torch

from assetgen.splats import Gaussians


def blob_gaussians(parts, colors, sigma, n_per=1500, seed=0):
    """Filled ellipsoids of small opaque Gaussians, one per ((center, radii), colour)."""
    rng = np.random.default_rng(seed)
    pts, cols = [], []
    for (c, radii), col in zip(parts, colors):
        d = rng.normal(size=(n_per, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        r = rng.uniform(0.0, 1.0, size=(n_per, 1)) ** (1 / 3)
        pts.append(np.asarray(c) + d * r * np.asarray(radii))
        cols.append(np.tile(col, (n_per, 1)))
    p = torch.tensor(np.concatenate(pts), dtype=torch.float32)
    n = len(p)
    return Gaussians(p, torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1), torch.full((n, 3), sigma),
                     torch.full((n,), 0.9), torch.tensor(np.concatenate(cols), dtype=torch.float32))


@pytest.fixture
def figure():
    """A small canonical character: Y up, facing +Z (red nose on +Z, green pack on -Z), left arm longer."""
    parts = [
        (((0, 0.55, 0), (0.12, 0.2, 0.08)), (0.3, 0.3, 0.8)),        # torso
        (((0, 0.85, 0), (0.08, 0.08, 0.08)), (0.9, 0.8, 0.6)),       # head
        (((0, 0.86, 0.08), (0.03, 0.03, 0.03)), (0.9, 0.1, 0.1)),    # nose
        (((-0.06, 0.18, 0), (0.04, 0.18, 0.04)), (0.2, 0.2, 0.2)),   # right leg
        (((0.06, 0.18, 0), (0.04, 0.18, 0.04)), (0.2, 0.2, 0.2)),    # left leg
        (((-0.2, 0.62, 0), (0.08, 0.03, 0.03)), (0.9, 0.8, 0.6)),    # right arm
        (((0.24, 0.62, 0), (0.12, 0.03, 0.03)), (0.9, 0.8, 0.6)),    # left arm
        (((0, 0.55, -0.1), (0.08, 0.12, 0.03)), (0.1, 0.6, 0.1)),    # backpack
    ]
    return blob_gaussians([p for p, _ in parts], [c for _, c in parts], sigma=0.012, n_per=1200)
