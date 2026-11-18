"""Stage 2: textured mesh -> skeleton + skin weights with SkinTokens (TokenRig).

Runs SkinTokens from its checkout without its demo, Gradio or Blender server:
the mesh goes in through SkinTokens' own .npz loader and transform, the rig
comes back in its normalised frame and is mapped onto our mesh exactly (the
transform is a uniform scale and shift of the same vertices), then written as
a skinned glTF with the original texture.

flash-attn is replaced by an SDPA shim and the Qwen3 backbone uses SDPA, so the
same code runs on ROCm, CUDA and CPU.

Run as ``python -m assetgen.rig`` (the orchestrator does) so the model's memory
is released when the stage ends.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

from .gltf import Rig, load_textured_glb, write_rigged_glb
from .thirdparty import SKINTOKENS, skintokens_dir

SHIMS = Path(__file__).resolve().parent / "_shims"


def weld(vertices: np.ndarray, faces: np.ndarray):
    """Merge vertices that share a position (UV seams). Returns welded verts, faces, inverse map."""
    uniq, inverse = np.unique(vertices, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    f = inverse[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return uniq, f[keep], inverse


def fit_scale_shift(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray]:
    """Least-squares s, t with dst = s * src + t."""
    sc, dc = src.mean(0), dst.mean(0)
    s = float(((src - sc) * (dst - dc)).sum() / ((src - sc) ** 2).sum())
    return s, dc - s * sc


def merge_small_leaves(joints, parents, names, weights, max_joints: int):
    """Fold leaf joints into their parents (lightest skin first) until at most ``max_joints`` remain."""
    joints, parents, names, weights = joints.copy(), parents.copy(), list(names), weights.copy()
    while len(joints) > max_joints:
        children = np.bincount(parents[parents >= 0], minlength=len(joints))
        leaves = np.nonzero((children == 0) & (parents >= 0))[0]
        j = leaves[np.argmin(weights[:, leaves].sum(0))]
        weights[:, parents[j]] += weights[:, j]
        keep = np.arange(len(joints)) != j
        remap = np.cumsum(keep) - 1
        parents = np.where(parents[keep] >= 0, remap[np.maximum(parents[keep], 0)], -1)
        joints, weights = joints[keep], weights[:, keep]
        names = [n for n, k in zip(names, keep) if k]
    return joints, parents, names, weights


def unique_names(names: list[str]) -> list[str]:
    seen, out = {}, []
    for n in names:
        n = n.replace(":", "_")
        k = seen.get(n, 0)
        out.append(n if k == 0 else f"{n}.{k:03d}")
        seen[n] = k + 1
    return out


def _install_runtime_patches():
    """SkinTokens hard-requires flash-attn, asks transformers for flash_attention_2, and imports
    diffusers' LayerNorm alias, which diffusers 0.43 removed (it was torch's LayerNorm on torch >= 2.1)."""
    sys.path.insert(0, str(SHIMS))
    import diffusers.models.normalization as normalization
    import torch
    import transformers

    if not hasattr(normalization, "LayerNorm"):
        normalization.LayerNorm = torch.nn.LayerNorm
    if not torch.cuda.is_available():
        # SkinTokens asks for the GPU's name at import time (to decide on FlashAttention 3).
        torch.cuda.get_device_name = lambda *args, **kwargs: "cpu"

    original = transformers.AutoModelForCausalLM.from_config

    def from_config(config, **kwargs):
        if kwargs.get("attn_implementation") == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"
        return original(config, **kwargs)

    transformers.AutoModelForCausalLM.from_config = from_config


def predict(mesh_path, device: str = "cuda", cls: str = "articulation", seed: int = 0, num_beams: int = 10,
            top_k: int = 5, top_p: float = 0.95, temperature: float = 1.0, repetition_penalty: float = 2.0,
            postprocess: bool = False, log=print):
    """Returns (TexturedMesh, Rig) with the rig in the mesh's own frame."""
    import torch

    mesh = load_textured_glb(mesh_path)
    verts, faces, inverse = weld(mesh.vertices.astype(np.float64), mesh.faces.astype(np.int64))
    log(f"[rig] {len(verts)} welded vertices, {len(faces)} faces")

    root = skintokens_dir()
    if not (root / SKINTOKENS["checkpoint"]).exists():
        raise SystemExit(f"SkinTokens is not set up at {root}: run `uv run assetgen setup`")
    os.chdir(root)
    sys.path.insert(0, str(root))
    _install_runtime_patches()

    from src.data.datapath import NpzLazyAsset
    from src.data.dataset import RigDataset
    from src.data.transform import Transform
    from src.data.vertex_group import voxel_skin
    from src.model.tokenrig import TokenRig
    from src.tokenizer.parse import get_tokenizer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = TokenRig.load_from_system_checkpoint(checkpoint_path=SKINTOKENS["checkpoint"]).to(device).eval()
    tokenizer = get_tokenizer(**model.tokenizer_config)
    transform = Transform.parse(**model.transform_config["predict_transform"])
    if cls not in tokenizer.cls_token_id:
        raise ValueError(f"unknown rig class {cls!r}; the model knows {sorted(tokenizer.cls_token_id)}")

    with tempfile.TemporaryDirectory() as tmp:
        npz = Path(tmp) / "mesh.npz"
        np.savez(npz, vertices=verts.astype(np.float32), faces=faces.astype(np.int64))
        ds = RigDataset(data=[NpzLazyAsset(path=str(npz), cls=cls)], transform=transform,
                        tokenizer=tokenizer, process_fn=model._process_fn)
        batch = ds.collate_fn([ds[0]])
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch.pop("skeleton_tokens", None)
    batch.pop("skeleton_mask", None)
    batch["generate_kwargs"] = dict(max_length=2048, top_k=top_k, top_p=top_p, temperature=temperature,
                                    repetition_penalty=repetition_penalty, num_return_sequences=1,
                                    num_beams=num_beams, do_sample=True)
    log(f"[rig] generating on {device} (beams={num_beams}, class={cls})")
    with torch.no_grad():
        result = model.predict_step(batch, make_asset=True)["results"][0]
    asset = result.asset
    if postprocess:
        voxel = asset.voxel(resolution=196)
        asset.skin *= voxel_skin(grid=0, grid_coords=voxel.coords, joints=asset.joints,
                                 vertices=asset.vertices, faces=asset.faces, mode="square",
                                 voxel_size=voxel.voxel_size)
        asset.normalize_skin()

    if len(asset.vertices) != len(verts):
        raise RuntimeError(f"SkinTokens returned {len(asset.vertices)} vertices for {len(verts)} given")
    s, t = fit_scale_shift(np.asarray(asset.vertices, dtype=np.float64), verts)
    joints = s * np.asarray(asset.joints, dtype=np.float64) + t
    names = list(result.detokenize_output.joint_names or [f"bone_{i}" for i in range(len(joints))])
    weights = np.asarray(asset.skin, dtype=np.float64)[inverse]
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-8)
    rig = Rig(joints=joints.astype(np.float32), parents=np.asarray(asset.parents, dtype=np.int64),
              names=unique_names(names), weights=weights.astype(np.float32))
    log(f"[rig] {len(joints)} joints, class {result.detokenize_output.cls or 'none'}")
    return mesh, rig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cls", default="articulation",
                    help="rig class the model is conditioned on: articulation (any shape), or a template such as mixamo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_beams", type=int, default=10)
    ap.add_argument("--max_joints", type=int, default=70)
    ap.add_argument("--postprocess", action="store_true", help="SkinTokens' voxel skin clean-up")
    args = ap.parse_args(argv)
    from .device import use_device

    use_device(args.device)
    out = Path(args.out).resolve()
    mesh_path = Path(args.mesh).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh, rig = predict(mesh_path, args.device, args.cls, args.seed, args.num_beams, postprocess=args.postprocess)
    if len(rig.joints) > args.max_joints:
        j, p, n, w = merge_small_leaves(rig.joints, rig.parents, rig.names, rig.weights, args.max_joints)
        print(f"[rig] merged {len(rig.joints) - len(j)} leaf joints to fit {args.max_joints}")
        rig = Rig(j, p, n, w)
    write_rigged_glb(mesh, rig, out)
    from PIL import Image

    from .preview import rig_preview

    Image.fromarray(rig_preview(mesh.vertices, mesh.faces, mesh.uvs, mesh.texture, rig.joints, rig.parents)).save(
        out.with_name("preview.png"))
    (out.with_suffix(".json")).write_text(json.dumps({
        "joints": rig.joints.tolist(), "parents": rig.parents.tolist(), "names": rig.names}, indent=1))
    print(f"[rig] wrote {out}")


if __name__ == "__main__":
    main()
