import torch

from assetgen.splats import (Gaussians, axis_angle_matrix, load_ply, make_camera, matrix_to_quat, quat_to_matrix,
                             render, save_ply)


def two_discs(order):
    """A red disc at z=0.2 in front of a blue one at z=-0.2 (camera on +Z), in either list order."""
    means = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, -0.2]])
    colors = torch.tensor([[1.0, 0, 0], [0, 0, 1.0]])
    idx = torch.tensor(order)
    return Gaussians(means[idx], torch.tensor([[1.0, 0, 0, 0]] * 2), torch.full((2, 3), 0.1),
                     torch.tensor([0.99, 0.99]), colors[idx])


def test_front_to_back_order_independent_of_input_order():
    cam = make_camera((0, 0, 2), (0, 0, 0), 40, 64)
    for order in ([0, 1], [1, 0]):
        c = render(two_discs(order), cam).rgb[32, 32]
        assert c[0] > 0.9 and c[2] < 0.1, order


def reference_render(g, cam):
    """Per-pixel brute force: every Gaussian, one global depth sort, no tiles."""
    H, W = cam.height, cam.width
    R, t = cam.view[:3, :3], cam.view[:3, 3]
    mc = g.means @ R.T + t
    z = mc[:, 2]
    J = torch.zeros(len(z), 2, 3)
    J[:, 0, 0] = cam.fx / z
    J[:, 0, 2] = -cam.fx * mc[:, 0] / z ** 2
    J[:, 1, 1] = cam.fy / z
    J[:, 1, 2] = -cam.fy * mc[:, 1] / z ** 2
    cov = J @ (R @ g.covariances() @ R.T) @ J.transpose(1, 2) + 0.3 * torch.eye(2)
    inv = torch.linalg.inv(cov)
    uv = torch.stack([cam.fx * mc[:, 0] / z + cam.cx, cam.fy * mc[:, 1] / z + cam.cy], 1)
    ys, xs = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
    d = torch.stack([xs, ys], -1).reshape(-1, 1, 2) - uv[None]
    power = -0.5 * torch.einsum("pni,nij,pnj->pn", d, inv, d)
    a = (g.opacities[None] * torch.exp(power)).clamp_max(0.99)
    a = torch.where(a >= 1 / 255, a, torch.zeros(()))
    order = torch.argsort(z)
    a, col = a[:, order], g.colors[order]
    T = torch.cumprod(1 - a, 1)
    w = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], 1) * a
    return (w @ col).reshape(H, W, 3)


def test_matches_brute_force_reference():
    """Tiling, per-tile depth order and batching must not change the image."""
    gen = torch.Generator().manual_seed(0)
    n = 300
    q = torch.nn.functional.normalize(torch.randn(n, 4, generator=gen), dim=-1)
    g = Gaussians((torch.rand(n, 3, generator=gen) - 0.5) * 0.8, q, torch.rand(n, 3, generator=gen) * 0.05 + 0.01,
                  torch.rand(n, generator=gen) * 0.8 + 0.2, torch.rand(n, 3, generator=gen))
    cam = make_camera((0.3, 0.4, 1.8), (0, 0, 0), 45, 64)
    ours = render(g, cam, budget=1 << 14).rgb
    ref = reference_render(g, cam)
    assert (ours - ref).abs().max() < 2e-2
    assert (ours - ref).abs().mean() < 2e-3


def test_median_depth_is_front_surface():
    r = render(two_discs([1, 0]), make_camera((0, 0, 2), (0, 0, 0), 40, 64))
    assert abs(float(r.depth[32, 32]) - 1.8) < 0.02


def test_image_axes():
    g = Gaussians(torch.tensor([[0.0, 0.3, 0.0], [0.3, 0.0, 0.0]]), torch.tensor([[1.0, 0, 0, 0]] * 2),
                  torch.full((2, 3), 0.03), torch.ones(2), torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]))
    r = render(g, make_camera((0, 0, 2), (0, 0, 0), 40, 64))
    ys, _ = torch.nonzero(r.rgb[..., 0] > 0.5, as_tuple=True)
    _, xs = torch.nonzero(r.rgb[..., 1] > 0.5, as_tuple=True)
    assert ys.float().mean() < 28          # +Y is up in the image
    assert xs.float().mean() > 36          # +X is to the right, seen from +Z


def test_quaternion_is_wxyz():
    """A Gaussian long along its local x, turned 90 degrees about z, is long along world y."""
    R = axis_angle_matrix([0, 0, 1], 90)
    q = matrix_to_quat(R)
    assert torch.allclose(quat_to_matrix(q), R, atol=1e-5)
    g = Gaussians(torch.zeros(1, 3), q[None], torch.tensor([[0.2, 0.02, 0.02]]), torch.ones(1), torch.ones(1, 3))
    r = render(g, make_camera((0, 0, 2), (0, 0, 0), 40, 64))
    ys, xs = torch.nonzero(r.alpha > 0.5, as_tuple=True)
    assert (ys.max() - ys.min()) > 3 * (xs.max() - xs.min())


def test_ply_round_trip(tmp_path, figure):
    save_ply(figure, tmp_path / "f.ply")
    g = load_ply(tmp_path / "f.ply")
    for a, b in ((g.means, figure.means), (g.scales, figure.scales), (g.colors, figure.colors),
                 (g.opacities, figure.opacities), (g.quats, figure.quats)):
        assert torch.allclose(a, b, atol=1e-4)


def test_transformed_rotates_covariance():
    g = Gaussians(torch.zeros(1, 3), torch.tensor([[1.0, 0, 0, 0]]), torch.tensor([[0.3, 0.1, 0.05]]),
                  torch.ones(1), torch.ones(1, 3))
    R = axis_angle_matrix([0.3, 1.0, 0.2], 70)
    expected = R @ g.covariances()[0] @ R.T
    assert torch.allclose(g.transformed(R, torch.zeros(3)).covariances()[0], expected, atol=1e-6)
