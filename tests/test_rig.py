import numpy as np

from assetgen.animate import clip_name, facing_pair
from assetgen.rig import fit_scale_shift, merge_small_leaves, unique_names, weld


def test_weld_merges_seam_duplicates():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    f = np.array([[0, 1, 2], [3, 5, 4]])
    wv, wf, inv = weld(v, f)
    assert len(wv) == 4 and len(wf) == 2
    assert np.allclose(wv[inv], v)


def test_fit_scale_shift():
    src = np.random.default_rng(0).random((50, 3))
    s, t = fit_scale_shift(src, 2.5 * src + [1, -2, 3])
    assert np.isclose(s, 2.5) and np.allclose(t, [1, -2, 3])


def test_merge_small_leaves_keeps_a_valid_tree():
    parents = np.array([-1, 0, 1, 1, 0])
    w = np.random.default_rng(1).random((20, 5))
    w /= w.sum(1, keepdims=True)
    j, p, n, w2 = merge_small_leaves(np.zeros((5, 3)), parents, list("abcde"), w, 3)
    assert len(j) == 3 and len(n) == 3
    assert np.allclose(w2.sum(1), 1)
    assert p[0] == -1 and (p[1:] >= 0).all() and (p[1:] < len(p)).all()


def test_unique_names():
    assert unique_names(["mixamorig:Hips", "a", "a"]) == ["mixamorig_Hips", "a", "a.001"]


def test_facing_pair_picks_mirrored_hips_right_at_minus_x():
    names = ["root", "spine", "hip_l", "hip_r", "knee_l", "knee_r", "hand_l"]
    joints = np.array([[0, 1, 0], [0, 1.4, 0], [0.15, 0.95, 0], [-0.15, 0.95, 0],
                       [0.15, 0.5, 0.05], [-0.15, 0.5, 0.05], [0.7, 1.3, 0.2]])
    parents = np.array([-1, 0, 0, 0, 2, 3, 1])
    assert facing_pair(joints, parents, names) == ("hip_r", "hip_l", False)


def test_facing_pair_falls_back_to_head_tail():
    joints = np.array([[0, 0.1, z] for z in np.linspace(-1, 1, 6)])
    right, left, axis = facing_pair(joints, np.array([-1, 0, 1, 2, 3, 4]), [f"b{i}" for i in range(6)])
    assert axis and right == "b5" and left == "b0"


def test_clip_name():
    assert clip_name("An object walks forward.") == "walks_forward"


def test_run_fetches_missing_third_party_once(monkeypatch):
    from assetgen import thirdparty

    calls = []
    monkeypatch.setattr(thirdparty, "setup", lambda log=print: calls.append("setup"))
    monkeypatch.setattr(thirdparty, "missing", lambda: ["SkinTokens checkout"])
    thirdparty.require(log=lambda *_: None)
    monkeypatch.setattr(thirdparty, "missing", lambda: [])
    thirdparty.require(log=lambda *_: None)
    assert calls == ["setup"]


def test_score_prefers_branched_symmetric_rig_over_a_chain():
    import trimesh

    from assetgen.gltf import Rig
    from assetgen.rig import longest_chain, score_rig

    box = trimesh.creation.box(extents=(1.0, 2.0, 0.6))
    v, f = np.asarray(box.vertices), np.asarray(box.faces)
    # hips, spine, chest, head, then left/right legs and arms, all mirrored across x = 0
    joints = np.array([[0, 0, 0], [0, 0.3, 0], [0, 0.6, 0], [0, 0.9, 0],
                       [0.2, -0.1, 0], [0.2, -0.6, 0], [-0.2, -0.1, 0], [-0.2, -0.6, 0],
                       [0.35, 0.6, 0], [0.45, 0.2, 0], [-0.35, 0.6, 0], [-0.45, 0.2, 0]], float)
    parents = np.array([-1, 0, 1, 2, 0, 4, 0, 6, 2, 8, 2, 10])
    chain = np.c_[np.full(30, 0.1), np.linspace(-0.9, 0.9, 30), np.zeros(30)]
    w = np.ones((len(v), 1))
    good = score_rig(Rig(joints, parents, [f"b{i}" for i in range(12)], w), v, f)
    bad = score_rig(Rig(chain, np.arange(-1, 29), [f"c{i}" for i in range(30)], w), v, f)
    assert longest_chain(np.arange(-1, 29)) == 30
    assert good["score"] > bad["score"]
    assert good["symmetry"] < 1e-6
