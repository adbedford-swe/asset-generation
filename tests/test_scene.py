import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from assetgen import export
from assetgen.mesh import export_glb
from assetgen.scene import SpecError, parse
from assetgen.scene import harmonize, terrain
from assetgen.scene.layout import Footprint, plan
from tests.test_prop import box, textured


def spec_of(**changes):
    data = {"name": "camp", "seed": 3, "ground": {"size": [24, 20]},
            "objects": [{"id": "campfire", "prompt": "a campfire", "height": 0.4, "place": [[0, 0]]},
                        {"id": "tent", "prompt": "a tent", "height": 2.0,
                         "place": [{"position": [3, -2], "yaw": {"face": "campfire"}}]},
                        {"id": "pine", "prompt": "a pine", "height": 8, "scatter": {"count": 12, "region": "edge"}},
                        {"id": "rock", "prompt": "a rock", "height": 0.6,
                         "scatter": {"count": 10, "tilt": True, "scale": [0.6, 1.4]}}]}
    data.update(changes)
    return data


FOOTPRINTS = {"campfire": Footprint(0.4, 0.55, 0.55), "tent": Footprint(2.0, 1.75, 1.6),
              "pine": Footprint(8.0, 2.3, 0.25), "rock": Footprint(0.6, 0.7, 0.7)}


@pytest.mark.parametrize("change, message", [
    ({"objects": []}, "at least one object"),
    ({"objects": [{"id": "x", "height": 1, "place": [[0, 0]]}]}, "give a prompt"),
    ({"objects": [{"id": "x", "prompt": "x", "place": [[0, 0]]}]}, "height"),
    ({"objects": [{"id": "x", "prompt": "x", "height": 1}]}, "place or scatter"),
    ({"objects": [{"id": "x", "prompt": "x", "height": 1, "place": [[0, 0]], "colour": "red"}]}, "unknown key colour"),
    ({"objects": [{"id": "x", "prompt": "x", "height": 1, "place": [{"position": [0, 0], "yaw": {"face": "y"}}]}]},
     "not an object id"),
    ({"objects": [{"id": "x", "prompt": "x", "height": 1, "scatter": {"count": 0}}]}, "count"),
    ({"objects": [{"id": "1x", "prompt": "x", "height": 1, "place": [[0, 0]]}]}, "start with a letter"),
    ({"ground": {"size": [10]}}, "size"),
])
def test_spec_errors_say_what_is_wrong(change, message):
    with pytest.raises(SpecError, match=message):
        parse(spec_of(**change))


def test_spec_reads_paths_from_its_folder(tmp_path):
    (tmp_path / "camp.json").write_text(json.dumps(spec_of(objects=[
        {"id": "barrel", "image": "pics/barrel.png", "height": 1, "redraw": 0.4, "place": [[0, 0]]}])))
    from assetgen.scene import load

    spec = load(tmp_path / "camp.json")
    assert spec.objects[0].image == str(tmp_path / "pics" / "barrel.png") and spec.objects[0].source == "image"
    assert spec.to_dict()["objects"][0]["redraw"] == 0.4


def test_tileable_textures_hide_the_seam():
    rng = np.random.default_rng(0)
    gradient = np.linspace(0, 255, 128)[None, :, None] * np.ones((128, 1, 3))
    noisy = np.clip(gradient + rng.normal(0, 10, (128, 128, 3)), 0, 255).astype(np.uint8)
    assert terrain.seam_error(noisy) > 10
    assert terrain.seam_error(terrain.tileable(noisy)) < 2


def test_heightfield_relief_and_flat_pads():
    field = terrain.heightfield((30, 20), relief=2.0, feature=6, resolution=0.5, seed=1)
    assert field.heights.shape == (41, 61)
    assert field.heights.max() - field.heights.min() == pytest.approx(2.0)
    flat = terrain.flatten(field, [(5.0, 2.0, 2.0)])
    angles = np.linspace(0, 2 * np.pi, 16)
    ring = flat.height(5 + 1.5 * np.cos(angles), 2 + 1.5 * np.sin(angles))
    assert np.ptp(ring) < 0.02
    v, f, n, uv = terrain.ground_mesh(flat, tile=4.0)
    assert len(v) == 41 * 61 and len(f) == 2 * 40 * 60 and (n[:, 1] > 0).all()
    assert uv[:, 0].max() - uv[:, 0].min() == pytest.approx(30 / 4)
    e0, e1 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    assert (np.cross(e0, e1)[:, 1] > 0).all()  # faces point up


def test_layout_places_faces_and_spaces():
    spec = parse(spec_of(ground={"size": [24, 20], "relief": 1.5, "feature": 6}))
    ground = terrain.heightfield(spec.ground.size, 1.5, 6, 0.5, spec.seed)
    layout = plan(spec, FOOTPRINTS, ground)
    assert not layout.warnings
    by = {}
    for i in layout.instances:
        by.setdefault(i.object, []).append(i)
    assert [len(by[k]) for k in ("campfire", "tent", "pine", "rock")] == [1, 1, 12, 10]
    tent = by["tent"][0]
    assert tent.yaw == pytest.approx(math.degrees(math.atan2(-3, 2)) % 360, abs=0.01)  # its front faces the fire
    for pine in by["pine"]:
        x, _, z = pine.position
        assert abs(x) <= 12 - 2.3 * pine.scale + 1e-6 and abs(z) <= 10 - 2.3 * pine.scale + 1e-6
        assert min(12 - abs(x), 10 - abs(z)) < 6  # the edge band
    scattered = by["pine"] + by["rock"]
    for a in scattered:
        for b in (by["campfire"][0], tent):
            need = FOOTPRINTS[b.object].radius + 0.5 + FOOTPRINTS[a.object].base_radius * a.scale
            assert math.dist(a.position[::2], b.position[::2]) >= need - 1e-6
    for k, a in enumerate(scattered):
        for b in scattered[k + 1:]:
            ra = 0.5 * (FOOTPRINTS[a.object].radius + FOOTPRINTS[a.object].base_radius) * a.scale
            rb = 0.5 * (FOOTPRINTS[b.object].radius + FOOTPRINTS[b.object].base_radius) * b.scale
            assert math.dist(a.position[::2], b.position[::2]) >= min(ra, rb) - 1e-6
    for i in by["pine"]:  # upright, standing on the lowest ground under the trunk
        assert i.rotation[0] == pytest.approx(0) and i.rotation[2] == pytest.approx(0)
        x, y, z = i.position
        assert y <= layout.ground.height(x, z) + 1e-6
    assert any(abs(r.rotation[0]) + abs(r.rotation[2]) > 1e-3 for r in by["rock"])  # rocks lean with the slope
    assert layout.instances == plan(spec, FOOTPRINTS, ground).instances  # same seed, same scene


def test_crowded_scatter_says_so():
    spec = parse(spec_of(ground={"size": [6, 6]}))
    layout = plan(spec, FOOTPRINTS, terrain.heightfield((6, 6)))
    assert any(w.startswith("pine: room for") for w in layout.warnings)


def test_harmonize_matches_lighting_and_keeps_hue():
    from skimage.color import rgb2lab

    dark_red = np.zeros((32, 32, 3), np.uint8)
    dark_red[..., 0], dark_red[..., 1], dark_red[..., 2] = 110, 20, 20
    dark_red[:16] = (60, 60, 64)
    bright = harmonize.look(np.full((100, 3), [0.8, 0.8, 0.75]))
    own = harmonize.image_look(dark_red)
    out = harmonize.apply(dark_red, own, bright, 1.0)
    assert harmonize.image_look(out).lightness > own.lightness + 20
    hue = lambda rgb: math.degrees(math.atan2(*rgb2lab(rgb[None, None] / 255.0)[0, 0, [2, 1]]))  # noqa: E731
    assert hue(out[20, 5].astype(float)) == pytest.approx(hue(dark_red[20, 5].astype(float)), abs=8)
    assert np.array_equal(harmonize.apply(dark_red, own, bright, 0.0), dark_red)


@pytest.fixture
def built(tmp_path):
    from assetgen.scene.build import build

    lib = tmp_path / "lib"
    lib.mkdir()

    def make(name, parts, colour):
        v, f = [], []
        for lo, hi in parts:
            bv, bf = box(lo, hi)
            f += [[a + len(v), b + len(v), c + len(v)] for a, b, c in bf]
            v += bv
        m = textured(v, f)
        m.texture[...] = colour
        export_glb(m, lib / f"{name}.glb")
        return str(lib / f"{name}.glb")

    Image.fromarray((np.random.default_rng(0).random((32, 32, 3)) * 80 + 60).astype(np.uint8)).save(lib / "g.png")
    spec = parse(spec_of(
        ground={"size": [24, 20], "texture": str(lib / "g.png"), "relief": 1.0},
        objects=[{"id": "campfire", "asset": make("fire", [((-.4, 0, -.4), (.4, .3, .4))], (220, 90, 20)),
                  "height": 0.4, "place": [[0, 0]]},
                 {"id": "pine", "asset": make("tree", [((-.1, 0, -.1), (.1, 2, .1)), ((-1, 2, -1), (1, 5, 1))],
                                              (40, 120, 40)), "height": 8, "scatter": {"count": 6}}]))
    return build(spec, tmp_path / "out", devices=["cpu"], log=lambda *_: None)


def test_build_writes_one_instanced_gltf_and_the_scene_file(built):
    import pygltflib

    data = json.loads(built.read_text())
    assert data["format"] == "assetgen.scene" and data["name"] == "camp"
    assert [(o["id"], o["count"], o["collider"]) for o in data["objects"]] == [("campfire", 1, "convex"),
                                                                               ("pine", 6, "convex")]
    doc = pygltflib.GLTF2.load_binary(str(built.with_suffix(".glb")))
    assert sorted(m.name for m in doc.meshes) == ["campfire", "ground", "pine"]  # each mesh stored once
    nodes = {n.name for n in doc.nodes}
    assert {"ground", "campfire_0", *(f"pine_{k}" for k in range(6))} <= nodes
    assert {i["node"] for i in data["instances"]} <= nodes
    folder = built.parent
    assert (folder / "camp.png").exists() and (folder / "assets" / "pine.assetgen").exists()
    pine = json.loads((folder / "assets" / "pine.assetgen").read_text())
    assert pine["model"]["file"] == "pine.glb" and pine["model"]["height"] == pytest.approx(8.0, abs=1e-3)
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(data, json.loads((export.ROOT / "schema" / "scene.schema.json").read_text()))


def test_rebuilding_reuses_the_objects(built):
    from assetgen.scene import load
    from assetgen.scene.build import build

    folder = built.parent
    stamp = (folder / "props" / "pine" / "pine.glb").stat().st_mtime_ns
    spec = load(folder / "scene.json")
    spec.object("pine").scatter.count = 3
    data = json.loads(build(spec, folder, devices=["cpu"], log=lambda *_: None).read_text())
    assert sum(i["object"] == "pine" for i in data["instances"]) == 3
    assert (folder / "props" / "pine" / "pine.glb").stat().st_mtime_ns == stamp


@pytest.mark.parametrize("engine", ["godot", "unity"])
def test_export_a_scene(built, engine, tmp_path):
    import zipfile

    zip_path = export.export_zip(built.parent, engine, tmp_path / f"camp-{engine}.zip")
    names = set(zipfile.ZipFile(zip_path).namelist())
    assert {"camp/camp.glb", "camp/camp.assetgen", "camp/assets/pine.glb", "camp/assets/pine.assetgen"} <= names
    assert ("camp/camp.glb.import" in names) == (engine == "godot")
    assert ("camp/assets/pine.glb.import" in names) == (engine == "godot")


def test_example_spec_is_valid_against_its_schema():
    from assetgen.scene.plan import EXAMPLE

    parse(EXAMPLE)
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(EXAMPLE, json.loads((export.ROOT / "schema" / "scene-spec.schema.json").read_text()))


class StubChat(BaseHTTPRequestHandler):
    answers: list = []
    requests: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubChat.requests.append(body)
        if "response_format" in body:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "response_format is not supported"}')
            return
        content = StubChat.answers.pop(0)
        data = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_plan_retries_until_the_spec_holds():
    from assetgen.scene.plan import EXAMPLE
    from assetgen.scene.plan import plan as plan_scene

    broken = dict(EXAMPLE, objects=[dict(EXAMPLE["objects"][0], height=None)])
    StubChat.answers = ["Here you go:\n```json\n" + json.dumps(broken) + "\n```", json.dumps(EXAMPLE)]
    StubChat.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubChat)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        data = plan_scene("a forest camp", f"http://127.0.0.1:{server.server_address[1]}", size=24,
                          log=lambda *_: None)
    finally:
        server.shutdown()
    assert data == EXAMPLE
    first, retry = StubChat.requests[1], StubChat.requests[2]  # the first asked for JSON mode and was refused
    assert "24 by 24 metres" in first["messages"][1]["content"] and "response_format" not in first
    assert "height" in retry["messages"][-1]["content"]


def test_placed_objects_that_overlap_are_moved_apart():
    spec = parse(spec_of(objects=[
        {"id": "boat", "prompt": "a boat", "height": 1.5, "place": [[0, 0]]},
        {"id": "crate", "prompt": "a crate", "height": 0.8, "place": [[0, 0.5], [0, 0.5, 0.8]]},
        {"id": "tent", "prompt": "a tent", "height": 2.0, "place": [[0.2, -0.3]]}]))
    footprints = {"boat": Footprint(1.5, 1.8, 1.6), "crate": Footprint(0.8, 0.5, 0.5),
                  "tent": Footprint(2.0, 1.75, 1.6)}
    layout = plan(spec, footprints, terrain.heightfield(spec.ground.size))
    by = {i.node: i for i in layout.instances}
    ground_level = [by["boat_0"], by["crate_0"], by["tent_0"]]
    for k, a in enumerate(ground_level):
        for b in ground_level[k + 1:]:
            ra, rb = (max(footprints[i.object].base_radius, 0.5 * footprints[i.object].radius) for i in (a, b))
            assert math.dist(a.position[::2], b.position[::2]) >= ra + rb + 0.1 - 1e-3
    assert by["crate_1"].position == [0.0, 0.5, 0.8]  # given a height: stacked on purpose, left alone
    assert sum("so it does not overlap" in w for w in layout.warnings) == 3
