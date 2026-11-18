import pytest
from PIL import Image

from assetgen.orient import orient
from assetgen.synthetic import idol_like, input_image


@pytest.mark.parametrize("yaw,view", [(137.0, 30.0), (250.0, -35.0)])
def test_recovers_up_and_facing(tmp_path, figure, yaw, view):
    Image.fromarray(input_image(figure, view, size=256)).save(tmp_path / "in.png")
    g, o = orient(idol_like(figure, yaw), str(tmp_path / "in.png"), log=lambda *_: None)
    assert o.flipped
    assert abs((o.facing_yaw_deg - yaw + 180) % 360 - 180) < 3
    assert abs(float(g.means[:, 1].min())) < 0.05           # standing on y = 0
    nose = g.colors[:, 0] > 0.8
    assert float(g.means[nose, 2].mean()) > float(g.means[:, 2].mean())   # nose in front (+Z)
