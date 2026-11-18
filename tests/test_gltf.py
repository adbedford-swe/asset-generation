import numpy as np
import pygltflib as gl

from assetgen.finalize import append_accessor, merge_clips
from assetgen.gltf import Rig, write_rigged_glb
from assetgen.gltf_anim import AnimatedGLB
from assetgen.mesh import TexturedMesh


def column_asset():
    """A vertical strip skinned to a two-bone chain: below y=0.5 to 'root', above to 'upper'."""
    y = np.linspace(0, 1, 11)
    v = np.stack([np.r_[np.zeros(11), np.full(11, 0.1)], np.r_[y, y], np.zeros(22)], 1).astype(np.float32)
    f = np.array([[i, i + 1, i + 11] for i in range(10)] + [[i + 1, i + 12, i + 11] for i in range(10)], np.int32)
    w = np.zeros((22, 2), np.float32)
    w[:, 0] = v[:, 1] < 0.5
    w[:, 1] = 1 - w[:, 0]
    mesh = TexturedMesh(v, f, np.zeros((22, 2), np.float32), np.full((4, 4, 3), 128, np.uint8))
    rig = Rig(np.array([[0, 0, 0], [0, 0.5, 0]], np.float32), np.array([-1, 0]), ["root", "upper"], w)
    return mesh, rig


def add_rotation_clip(path, angle):
    """Turn the 'upper' bone about z by ``angle`` radians over one second."""
    doc = gl.GLTF2.load_binary(str(path))
    blob = bytearray(doc.binary_blob())
    node = next(i for i, n in enumerate(doc.nodes) if n.name == "upper")
    t = np.array([[0.0], [1.0]], np.float32)
    q = np.array([[0, 0, 0, 1], [0, 0, np.sin(angle / 2), np.cos(angle / 2)]], np.float32)
    ti = append_accessor(doc, blob, t, gl.Accessor(componentType=gl.FLOAT, type=gl.SCALAR, min=[0.0], max=[1.0]))
    qi = append_accessor(doc, blob, q, gl.Accessor(componentType=gl.FLOAT, type=gl.VEC4))
    doc.animations = [gl.Animation(name="spin", samplers=[gl.AnimationSampler(input=ti, output=qi)],
                                   channels=[gl.AnimationChannel(sampler=0, target=gl.AnimationChannelTarget(
                                       node=node, path="rotation"))])]
    doc.buffers[0].byteLength = len(blob)
    doc.set_binary_blob(bytes(blob))
    doc.save_binary(str(path))


def test_rigged_rest_pose_reproduces_mesh(tmp_path):
    mesh, rig = column_asset()
    write_rigged_glb(mesh, rig, tmp_path / "r.glb")
    g = AnimatedGLB(tmp_path / "r.glb")
    assert np.allclose(g.posed(None), mesh.vertices, atol=1e-5)
    assert np.allclose(g.joint_positions(None), rig.joints, atol=1e-6)


def test_merge_clips_retargets_names_and_scales(tmp_path):
    mesh, rig = column_asset()
    for name, angle in (("a", np.pi / 2), ("b", -np.pi / 2)):
        write_rigged_glb(mesh, rig, tmp_path / f"{name}.glb")
        add_rotation_clip(tmp_path / f"{name}.glb", angle)
    clips = [{"glb": str(tmp_path / "a.glb"), "prompt": "An object waves left."},
             {"glb": str(tmp_path / "b.glb"), "prompt": "An object waves right."}]
    info = merge_clips(clips, tmp_path / "out.glb", height=2.0, log=lambda *_: None)
    assert info["clips"] == ["waves_left", "waves_right"]
    g = AnimatedGLB(tmp_path / "out.glb")
    # The top vertex (0, 1, 0) turns +/-90 degrees about the joint at y = 0.5; everything doubled.
    assert np.allclose(g.posed(0, 1.0)[10], np.array([-0.5, 0.5, 0]) * 2.0, atol=1e-4)
    assert np.allclose(g.posed(1, 1.0)[10], np.array([0.5, 0.5, 0]) * 2.0, atol=1e-4)
