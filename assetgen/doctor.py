"""Check that everything a run needs is in place, before spending GPU time on it."""
from __future__ import annotations

import os
import shutil
import subprocess


def doctor() -> bool:
    ok = True

    def check(name, passed, detail="", fatal=True):
        nonlocal ok
        mark = "ok  " if passed else ("FAIL" if fatal else "warn")
        print(f"[{mark}] {name}{': ' + detail if detail else ''}")
        if fatal and not passed:
            ok = False

    import sys

    check("python", sys.version_info[:2] == (3, 11), sys.version.split()[0])

    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    import torch

    hip = getattr(torch.version, "hip", None)
    check("torch", True, f"{torch.__version__} ({'ROCm ' + hip if hip else 'CUDA ' + str(torch.version.cuda)})")
    check("GPU", torch.cuda.is_available(), "torch.cuda.is_available()")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            arch = getattr(p, "gcnArchName", "")
            check(f"cuda:{i}", True, f"{p.name} {arch} {p.total_memory / 2**30:.1f} GB")
        q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
            check("fused attention", True)
        except Exception as e:
            check("fused attention", False, f"falls back to the slow math kernel ({type(e).__name__})", fatal=False)

    for mod in ("bpy", "idol", "transformers", "diffusers", "torch_geometric", "spacy", "clip", "Animation",
                "xatlas", "fast_simplification", "skimage", "trimesh", "pygltflib"):
        try:
            __import__(mod)
            check(f"import {mod}", True)
        except Exception as e:
            check(f"import {mod}", False, f"{type(e).__name__}: {e}")

    from .thirdparty import SKINTOKENS, UNIMATE, TEXT_MODELS, unimate_exp_dir

    for spec in (SKINTOKENS, UNIMATE):
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=spec["dir"], capture_output=True, text=True).stdout.strip() \
            if (spec["dir"] / ".git").exists() else ""
        check(f"{spec['dir'].name} checkout", head == spec["commit"], head[:12] or "missing (run `assetgen setup`)")
    for f in SKINTOKENS["files"]:
        check(f"SkinTokens {f.split('/')[-2]}", (SKINTOKENS["dir"] / f).exists())
    check("Qwen3-0.6B config", (SKINTOKENS["dir"] / "models" / "Qwen3-0.6B" / "config.json").exists())
    for f in UNIMATE["files"]:
        check(f"UniMate {f.split('/')[-1]}", (UNIMATE["dir"] / "outputs" / f).exists())

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    from .generate import ENCODERS

    for repo in [*TEXT_MODELS, "speridlabs/idol", *ENCODERS.values()]:
        try:
            snapshot_download(repo, local_files_only=True,
                              allow_patterns=["vae/*"] if "FLUX" in repo else None)
            check(f"cached {repo}", True)
        except LocalEntryNotFoundError:
            gated = repo in ENCODERS.values()
            check(f"cached {repo}", False, "gated: request access and `hf auth login`" if gated else
                  "downloads on first use", fatal=False)

    free = shutil.disk_usage(unimate_exp_dir().parent if unimate_exp_dir().parent.exists() else ".").free / 2**30
    check("disk space", free > 10, f"{free:.0f} GB free", fatal=False)
    print("ready" if ok else "not ready")
    return ok
