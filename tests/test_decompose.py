import json
import math

import numpy as np
import pytest
import torch

from assetgen.preview import render_mesh
from assetgen.scene import parse
from assetgen.scene.decompose import Found, cutout, fit_ground, locate, overlay, to_spec
from assetgen.scene.terrain import ground_mesh, heightfield
from assetgen.splats import make_camera
from tests.test_prop import box

# Boxes on the ground (y = 0): label, centre (x, z), width, height, depth.
BOXES = [("crate", (0.0, 0.0), 1.0, 1.0, 1.0), ("pillar", (3.0, -2.0), 0.6, 3.0, 0.6),
         ("crate", (-2.5, 1.5), 1.0, 1.0, 1.0), ("wall", (-1.0, -4.0), 4.0, 2.0, 0.4)]
FOV = 50.0


def render_scene(size=480):
    """Colour, depth and per-box visible masks of the boxes on a big ground, seen from above at three quarters."""
    cam = make_camera(torch.tensor([7.0, 6.5, 9.0]), torch.tensor([0.0, 0.5, -1.0]), FOV, size)
    gv, gf, _, _ = ground_mesh(heightfield((40, 40), resolution=0.5), tile=4.0)
    _, _, depth = render_mesh(gv, gf, cam, vertex_colors=np.full((len(gv), 3), 0.5), return_depth=True)
    rgb = np.full((size, size, 3), 0.5)
    owner = np.full((size, size), -1)
    for k, (_, (x, z), w, h, d) in enumerate(BOXES):
        v, f = box((x - w / 2, 0, z - d / 2), (x + w / 2, h, z + d / 2))
        colour = np.tile([[0.2 + 0.2 * k, 0.3, 0.8 - 0.15 * k]], (8, 1))
        r, m, dz = render_mesh(np.array(v, np.float32), np.array(f), cam, vertex_colors=colour, return_depth=True)
        nearer = (m > 0.5) & (dz < depth)
        depth = torch.where(nearer, dz, depth)
        rgb[nearer.numpy()] = r.numpy()[nearer.numpy()]
        owner[nearer.numpy()] = k
    return (rgb * 255).astype(np.uint8), depth.numpy().astype(np.float64), owner, cam


@pytest.fixture(scope="module")
def scene():
    image, depth, owner, cam = render_scene()
    found = [Found(label, 0.9, [0, 0, 1, 1], owner == k) for k, (label, *_) in enumerate(BOXES)]
    placed, ground = locate(found, depth, FOV)
    return image, placed, ground


def test_the_ground_plane_is_found():
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(-5, 5, 3000), np.zeros(3000), rng.uniform(4, 20, 3000)]
    tilt = np.array([[1, 0, 0], [0, math.cos(0.6), -math.sin(0.6)], [0, math.sin(0.6), math.cos(0.6)]])
    pts = pts @ tilt.T + [0, 2.0, 0]
    pts = np.vstack([pts, rng.uniform(-5, 5, (300, 3)) + [0, -2, 10]])  # clutter above the ground
    n, d = fit_ground(pts)
    assert abs(n @ (tilt @ [0, -1, 0])) == pytest.approx(1, abs=1e-3)
    assert np.abs(pts[:3000] @ n + d).max() < 1e-3


def test_objects_stand_where_they_were(scene):
    _, placed, ground = scene
    assert ground.camera_height > 0
    for p, (_, _, w, h, _) in zip(placed, BOXES):
        assert p.height == pytest.approx(h, rel=0.06)
    # Distances between bases are what they were (the world frame's origin and turn are the picture's own).
    for a, (_, ca, *_) in zip(placed, BOXES):
        for b, (_, cb, *_) in zip(placed, BOXES):
            assert math.dist(a.position, b.position) == pytest.approx(math.dist(ca, cb), abs=0.35)
    # Every object was seen from the camera's side: the direction to the camera is +z in the picture's frame.
    for p in placed:
        assert abs((p.seen_from + 180) % 360 - 180) < 45


def test_hidden_objects_count_as_less_visible(scene):
    _, placed, _ = scene
    assert all(0 <= p.visible <= 1 for p in placed)


def test_spec_groups_labels_and_cuts_out(scene, tmp_path):
    image, placed, _ = scene
    spec = to_spec(placed, image, tmp_path / "parts", "yard", style="weathered", scale=2.0)
    ids = [o["id"] for o in spec["objects"]]
    assert ids == ["crate", "pillar", "wall"]
    crate = spec["objects"][0]
    assert len(crate["place"]) == 2 and crate["height"] == pytest.approx(2.0, rel=0.08)
    assert all("seen_from" in p["yaw"] for p in crate["place"])
    assert (tmp_path / "parts" / "crate.png").exists() and crate["image"] == "parts/crate.png"
    from PIL import Image

    cut = Image.open(tmp_path / "parts" / "crate.png")
    assert cut.mode == "RGBA" and cut.width == cut.height
    alpha = np.asarray(cut)[..., 3]
    assert alpha[0, 0] == 0 and alpha.max() == 255
    parse(spec, tmp_path)  # a spec the builder takes
    each = to_spec(placed, image, tmp_path / "each", "yard", each=True)
    assert len(each["objects"]) == len(placed)


def test_cutout_pads_at_the_picture_edge():
    image = np.zeros((40, 40, 3), np.uint8)
    mask = np.zeros((40, 40), bool)
    mask[0:10, 30:40] = True
    cut = np.asarray(cutout(image, mask))
    assert cut.shape[0] == cut.shape[1] and cut[..., 3].sum() > 0


def test_overlay_draws(scene):
    image, placed, ground = scene
    out = overlay(image, placed, ground, FOV)
    assert out.shape == image.shape and not np.array_equal(out, image)
    json.dumps([p.position for p in placed])


def test_the_ground_is_seen_from_above(scene):
    """Ground coloured by its own (x, z) comes back at the right places once the two frames are aligned."""
    from assetgen.scene.decompose import ground_texture

    _, placed, ground = scene
    image, depth, owner, cam = render_scene_coloured()
    texture, shown = ground_texture(image, owner >= 0, ground, FOV, (12, 12), pixels=96)
    assert 0.3 < shown <= 1
    # The picture's frame against the scene's: a turn and a shift that take the found bases onto the true ones.
    found = np.array([p.position for p in placed])
    true = np.array([c for _, c, *_ in BOXES])
    fc, tc = found.mean(0), true.mean(0)
    u, _, vt = np.linalg.svd((found - fc).T @ (true - tc))
    turn = (u @ vt).T
    xs = (np.arange(96) + 0.5) / 96 * 12 - 6
    X, Z = np.meshgrid(xs, xs)
    original = (np.stack([X, Z], -1) - fc) @ turn.T + tc
    expected = np.clip((original + 15) / 30 * 255, 0, 255)
    near = (np.abs(original) < 6).all(-1)
    error = np.abs(texture[..., :2].astype(float) - expected)[near]
    assert np.median(error) < 12


def render_scene_coloured(size=480):
    """render_scene, with the ground's red and green rising along x and z."""
    image, depth, owner, cam = render_scene(size)
    gv, gf, _, _ = ground_mesh(heightfield((40, 40), resolution=0.5), tile=4.0)
    colours = np.stack([(gv[:, 0] + 15) / 30, (gv[:, 2] + 15) / 30, np.full(len(gv), 0.3)], -1).clip(0, 1)
    rgb, mask, _ = render_mesh(gv, gf, cam, vertex_colors=colours, shade=False, return_depth=True)
    out = image.copy()
    out[owner < 0] = (rgb.numpy() * 255).astype(np.uint8)[owner < 0]
    return out, depth, owner, cam


def test_seen_from_turns_the_side_the_picture_showed():
    from assetgen.scene import terrain
    from assetgen.scene.layout import Footprint, plan

    spec = parse({"name": "s", "objects": [{"id": "cart", "prompt": "a cart", "height": 1,
                                            "place": [{"position": [0, 0], "yaw": {"seen_from": 30}}]}]})
    flat = terrain.heightfield(spec.ground.size)
    # The cart's own picture showed it from 90 degrees to its right; the scene saw it from 30 degrees.
    turned = plan(spec, {"cart": Footprint(1, 1, 1, view=90)}, flat).instances[0]
    assert turned.yaw == pytest.approx(300)
    assert plan(spec, {"cart": Footprint(1, 1, 1)}, flat).instances[0].yaw == pytest.approx(30)


def test_view_comes_from_the_mesh_stage(tmp_path):
    from assetgen.scene.build import view_of

    (tmp_path / "1_mesh").mkdir()
    (tmp_path / "1_mesh" / "orientation.json").write_text(json.dumps({"camera_yaw_deg": 100.0,
                                                                      "facing_yaw_deg": 130.0}))
    assert view_of(tmp_path) == pytest.approx(330)
    assert view_of(tmp_path / "missing") == 0


def test_decompose_writes_a_buildable_spec(scene, tmp_path, monkeypatch):
    from assetgen.scene import decompose as dec

    image, placed, ground = scene
    monkeypatch.setattr(dec, "analyse", lambda *a, **k: (placed, ground, image))
    from PIL import Image

    Image.fromarray(image).save(tmp_path / "yard.png")
    spec = dec.decompose(tmp_path / "yard.png", ["crate", "pillar", "wall"], tmp_path / "out" / "yard.json",
                         known={"pillar": 6.0}, log=lambda *_: None)
    assert spec["objects"][1]["height"] == pytest.approx(6.0, rel=1e-3)  # scaled to the known height
    assert spec["ground"]["texture"] == "yard_parts/ground.png" and spec["ground"]["tile"] == spec["ground"]["size"][0]
    folder = tmp_path / "out"
    assert (folder / "yard_found.png").exists() and (folder / "yard_concept.png").exists()
    assert (folder / "yard_parts" / "ground.png").exists()
    from assetgen.scene import load

    built = load(folder / "yard.json")
    assert built.ground.texture == str(folder / "yard_parts" / "ground.png")
    jsonschema = pytest.importorskip("jsonschema")
    from assetgen import export

    jsonschema.validate(spec, json.loads((export.ROOT / "schema" / "scene-spec.schema.json").read_text()))


def test_one_detection_per_object():
    from assetgen.scene.decompose import distinct

    barn = np.zeros((20, 20), bool)
    barn[2:15, 2:15] = True
    cart = np.zeros((20, 20), bool)
    cart[5:9, 5:9] = True  # on the barn's wall
    crate = np.zeros((20, 20), bool)
    crate[16:19, 16:19] = True
    found = [Found("cart", 0.4, [0, 0, 1, 1], cart), Found("barn", 0.7, [0, 0, 1, 1], barn),
             Found("crate", 0.3, [0, 0, 1, 1], crate)]
    kept, dropped = distinct(found)
    assert [f.label for f in kept] == ["barn", "crate"] and [f.label for f in dropped] == ["cart"]


def test_holes_fill_from_their_surroundings():
    from assetgen.scene.decompose import fill

    image = np.zeros((37, 50, 3))
    image[:, :25] = [200, 0, 0]
    image[:, 25:] = [0, 0, 200]
    known = np.ones((37, 50), bool)
    known[10:20, 5:15] = False   # inside the red half
    known[:, 45:] = False        # the blue edge
    out = fill(np.where(known[..., None], image, 0), known)
    assert np.abs(out[15, 10] - [200, 0, 0]).max() < 1
    assert np.abs(out[20, 48] - [0, 0, 200]).max() < 1


def test_patches_of_ground_are_told_from_objects(scene):
    from assetgen.scene.decompose import _is_sheet

    rng = np.random.default_rng(0)
    patch = np.c_[rng.uniform(0, 3, 2000), rng.normal(0, 0.03, 2000), rng.uniform(0, 2, 2000)]
    assert _is_sheet(patch)
    _, placed, _ = scene
    assert not any(p.sheet for p in placed)  # boxes show walls as well as tops
