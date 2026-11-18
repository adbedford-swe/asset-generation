import numpy as np
import torch

from assetgen.gltf import load_textured_glb
from assetgen.mesh import MeshSettings, TexturedMesh, export_glb, splat_to_mesh
from tests.conftest import blob_gaussians

FAST = MeshSettings(n_views=24, render_size=192, resolution=64, target_faces=4000, texture_size=256, smooth_iters=2)


def test_sphere_reconstruction_and_colours():
    g = blob_gaussians([((0, 0, 0), (0.3, 0.3, 0.3))], [(1.0, 0, 0)], sigma=0.015, n_per=12000)
    g.colors[g.means[:, 2] < 0] = torch.tensor([0.0, 0, 1.0])   # red front, blue back
    m = splat_to_mesh(g, FAST, log=lambda *_: None)
    r = np.linalg.norm(m.vertices, axis=1)
    assert abs(np.median(r) - 0.3) < 0.03
    assert len(m.faces) <= FAST.target_faces
    tex = m.texture.astype(float) / 255
    px = np.clip((m.uvs * FAST.texture_size).astype(int), 0, FAST.texture_size - 1)
    col = tex[px[:, 1], px[:, 0]]
    front, back = m.vertices[:, 2] > 0.1, m.vertices[:, 2] < -0.1
    assert col[front, 0].mean() > 0.7 and col[front, 2].mean() < 0.3
    assert col[back, 2].mean() > 0.7 and col[back, 0].mean() < 0.3


def test_glb_round_trip_is_exact(tmp_path):
    rng = np.random.default_rng(0)
    m = TexturedMesh(rng.random((30, 3)).astype(np.float32), rng.integers(0, 30, (40, 3)).astype(np.int32),
                     rng.random((30, 2)).astype(np.float32), rng.integers(0, 255, (16, 16, 3)).astype(np.uint8))
    export_glb(m, tmp_path / "m.glb")
    r = load_textured_glb(tmp_path / "m.glb")
    assert np.allclose(r.uvs, m.uvs, atol=1e-6)
    assert np.array_equal(r.texture, m.texture)
