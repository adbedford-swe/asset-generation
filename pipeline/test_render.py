"""Quick renderer smoke test: render a few orbit views of a .ply and save them."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from splat_render import load_gaussians, orbit_cameras, render_views

ply = sys.argv[1] if len(sys.argv) > 1 else "outputs/idol/fig-viking.ply"
out = Path("outputs/render_test")
device = "cuda:0"

t0 = time.time()
g = load_gaussians(ply, device=device)
print(f"[test] loaded {g['n']} gaussians in {time.time()-t0:.2f}s")

center = torch.zeros(3, device=device)
radius = 1.6
cams = orbit_cameras(center, radius, fov_deg=40.0, n_views=4, image_size=512, device=device)

t0 = time.time()
render_views(g, cams, out)
print(f"[test] rendered {len(cams)} views in {time.time()-t0:.2f}s -> {out}")
