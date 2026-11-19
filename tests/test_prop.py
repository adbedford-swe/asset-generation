import json
import zipfile

import numpy as np
import pytest

from assetgen import export, prop
from assetgen.mesh import TexturedMesh, export_glb
from assetgen.pipeline import Options, branch, kind_stages, stage_commands


def textured(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float32)
    uvs = (vertices[:, [0, 2]] - vertices[:, [0, 2]].min(0)) / np.ptp(vertices[:, [0, 2]], axis=0).clip(1e-6)
    texture = np.zeros((16, 16, 3), np.uint8)
    texture[..., 0] = 180
    return TexturedMesh(vertices, np.asarray(faces, dtype=np.int32), uvs.astype(np.float32), texture)


def box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
    f = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3],
         [1, 2, 6], [1, 6, 5]]
    return v, f


@pytest.fixture
def tree(tmp_path):
    """A 'tree' off-centre and floating: a thin trunk under a wide canopy."""
    tv, tf = box((-0.1, 0.5, -0.1), (0.1, 2.5, 0.1))
    cv, cf = box((-1.0, 2.5, -0.8), (1.0, 4.5, 0.8))
    v = np.array(tv + cv) + [3.0, 0.0, -2.0]
    f = np.array(tf + [[a + 8, b + 8, c + 8] for a, b, c in cf])
    path = tmp_path / "run" / "1_mesh" / "mesh.glb"
    path.parent.mkdir(parents=True)
    export_glb(textured(v, f), path)
    return path


def test_finalize_scales_grounds_and_measures(tree):
    out = tree.parent.parent / "pine.glb"
    data = json.loads(prop.finalize(tree, out, "prop", height=8.0, log=lambda *_: None).read_text())
    assert data["format"] == "assetgen.prop" and data["kind"] == "prop" and data["name"] == "pine"
    model = data["model"]
    assert model["height"] == pytest.approx(8.0, abs=1e-3)
    assert model["bounds"]["min"][1] == pytest.approx(0.0, abs=1e-4)
    assert model["bounds"]["min"][0] == pytest.approx(-model["bounds"]["max"][0], abs=1e-4)  # centred on x
    scale = 8.0 / 4.0
    assert model["size"] == pytest.approx([2.0 * scale, 8.0, 1.6 * scale], abs=1e-3)
    fp = data["footprint"]
    assert fp["radius"] == pytest.approx(np.hypot(1.0, 0.8) * scale, abs=1e-3)
    assert fp["base_radius"] == pytest.approx(np.hypot(0.1, 0.1) * scale, abs=1e-3)  # the trunk, not the canopy
    assert len(fp["outline"]) >= 4
    assert data["collider"]["type"] == "convex"
    assert data["collider"]["box"]["size"] == pytest.approx(model["size"], abs=1e-3)
    assert out.exists() and out.with_suffix(".png").exists() and data["preview"] == "pine.png"

    import trimesh

    scene = trimesh.load(str(out), force="scene")
    assert list(scene.graph.nodes_geometry) == ["pine"]


def test_dioramas_collide_with_their_mesh(tree, tmp_path):
    data = json.loads(prop.finalize(tree, tmp_path / "ruins.glb", "diorama", log=lambda *_: None).read_text())
    assert data["kind"] == "diorama" and data["collider"]["type"] == "mesh"
    assert data["model"]["height"] == pytest.approx(4.0, abs=1e-3)  # no height given: as meshed
    with pytest.raises(ValueError):
        prop.collider_for("prop", "sphere", np.zeros((3, 3)))


def test_outline_keeps_the_shape_with_fewer_corners():
    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    poly = prop.outline(circle, max_points=12)
    assert len(poly) == 12
    assert np.linalg.norm(poly, axis=1) == pytest.approx(np.ones(12), abs=1e-6)


def test_schema_matches_a_written_file(tree):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((export.ROOT / "schema" / "prop.schema.json").read_text())
    path = prop.finalize(tree, tree.parent.parent / "pine.glb", height=2.0, log=lambda *_: None)
    jsonschema.validate(json.loads(path.read_text()), schema)


def test_props_skip_rig_and_motion(tmp_path):
    o = Options(out=tmp_path / "barrel", text="a wooden barrel", name="barrel", kind="prop", height=1.0)
    cmds = stage_commands(o)
    assert list(cmds) == ["image", "generate", "mesh", "finalize"]
    image_args = cmds["image"][0]
    assert image_args[image_args.index("--kind") + 1] == "prop"
    fin = cmds["finalize"][0]
    assert fin[0] == "assetgen.prop" and fin[fin.index("--height") + 1] == "1.0"
    assert [p.name for p in cmds["finalize"][1]] == ["barrel.glb", "barrel.assetgen"]
    # A character's image stage is unchanged, so earlier runs stay up to date.
    assert "--kind" not in stage_commands(Options(out=tmp_path / "hero", text="a hero"))["image"][0]
    assert kind_stages("diorama") == ["image", "generate", "mesh", "finalize"]
    with pytest.raises(ValueError):
        stage_commands(Options(out=tmp_path / "x", text="x", kind="vehicle"))


def test_branch_a_prop_from_finalize(tree, tmp_path):
    source = tree.parent.parent
    (source / "run.json").write_text(json.dumps({"name": "pine", "kind": "prop", "ply": "splat.ply"}))
    (source / "manifest.json").write_text(json.dumps({"stages": {"mesh": {"args": [], "ok": True}}}))
    target = branch(source, "finalize", tmp_path / "again")
    assert (target / "1_mesh" / "mesh.glb").exists()
    with pytest.raises(SystemExit, match="no rig stage"):
        branch(source, "rig", tmp_path / "rigged")


@pytest.mark.parametrize("engine", ["unity", "godot", "gltf"])
def test_export_a_prop(tree, engine, tmp_path):
    glb = tree.parent.parent / "pine.glb"
    prop.finalize(tree, glb, height=6.0, log=lambda *_: None)
    zip_path = export.export_zip(glb, engine, tmp_path / f"pine-{engine}.zip")
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        data = json.loads(z.read("pine/pine.assetgen"))
    assert {"pine/pine.glb", "pine/pine.assetgen", "pine/pine_preview.png"} <= names
    assert data["preview"] == "pine_preview.png" and data["format"] == "assetgen.prop"
    assert ("pine/pine.glb.import" in names) == (engine == "godot")
