"""Pinned third-party code and weights, and where they live.

SkinTokens and UniMate are research repositories rather than packages: they are
run from their own checkouts (both read checkpoints relative to their root), so
`assetgen setup` clones them at fixed commits into ``third_party/`` and puts the
weights where each expects them.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = Path(os.environ.get("ASSETGEN_THIRD_PARTY", ROOT / "third_party"))

SKINTOKENS = {
    "url": "https://github.com/VAST-AI-Research/SkinTokens.git",
    "commit": "273b691d35989d71cd17ff2895fdc735097b92d1",
    "dir": THIRD_PARTY / "SkinTokens",
    "repo_id": "VAST-AI/SkinTokens",
    "files": [
        "experiments/skin_vae_2_10_32768/last.ckpt",
        "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
    ],
    "checkpoint": "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
}

UNIMATE = {
    "url": "https://github.com/Friedrich-M/UniMate.git",
    "commit": "b78c7807df7cf4231d4e67f8d0ece29c5e11025b",
    "dir": THIRD_PARTY / "UniMate",
    "repo_id": "Linzhan/UniMate",
    "experiment": "unimate_uniml3d_f60_v3",
    "files": [
        "unimate_uniml3d_f60_v3/config.json",
        "unimate_uniml3d_f60_v3/dataset_stats.npy",
        "unimate_uniml3d_f60_v3/checkpoints/checkpoint_step_150000.pt",
    ],
}

# Point tracking for motion from video (assetgen.capture). Code and weights are CC BY-NC 4.0.
COTRACKER = {
    "url": "https://github.com/facebookresearch/co-tracker.git",
    "commit": "82e02e8029753ad4ef13cf06be7f4fc5facdda4d",
    "dir": THIRD_PARTY / "co-tracker",
    "repo_id": "facebook/cotracker3",
    "files": ["scaled_offline.pth"],
}

# Fetched ahead of time so a run never needs the network.
TEXT_MODELS = ["google/flan-t5-base"]
QWEN_CONFIG = "Qwen/Qwen3-0.6B"


def skintokens_dir() -> Path:
    return SKINTOKENS["dir"]


def unimate_dir() -> Path:
    return UNIMATE["dir"]


def unimate_exp_dir() -> Path:
    return UNIMATE["dir"] / "outputs" / UNIMATE["experiment"]


def cotracker_dir() -> Path:
    return COTRACKER["dir"]


def _git(*args, cwd=None):
    subprocess.run(["git", "-c", "core.longpaths=true", *args], cwd=cwd, check=True)


def checkout(spec: dict, log=print) -> None:
    """Shallow checkout of exactly the pinned commit."""
    path: Path = spec["dir"]
    if not (path / ".git").exists():
        log(f"[setup] fetching {spec['url']} @ {spec['commit'][:12]} -> {path}")
        path.mkdir(parents=True, exist_ok=True)
        _git("init", "--quiet", cwd=path)
        _git("remote", "add", "origin", spec["url"], cwd=path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True).stdout.strip()
    if head != spec["commit"]:
        _git("fetch", "--quiet", "--depth", "1", "origin", spec["commit"], cwd=path)
        _git("checkout", "--quiet", "--detach", spec["commit"], cwd=path)
    log(f"[setup] {path.name} at {spec['commit'][:12]}")


def download_weights(log=print) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    for f in SKINTOKENS["files"]:
        hf_hub_download(SKINTOKENS["repo_id"], f, local_dir=SKINTOKENS["dir"])
        log(f"[setup] SkinTokens {f}")
    snapshot_download(QWEN_CONFIG, local_dir=SKINTOKENS["dir"] / "models" / "Qwen3-0.6B",
                      ignore_patterns=["*.bin", "*.safetensors"])
    for f in UNIMATE["files"]:
        hf_hub_download(UNIMATE["repo_id"], f, local_dir=UNIMATE["dir"] / "outputs")
        log(f"[setup] UniMate {f}")
    for repo in TEXT_MODELS:
        snapshot_download(repo, allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"])
        log(f"[setup] {repo}")


def missing() -> list[str]:
    """What a run would need from ``setup`` that is not in place yet."""
    gaps = []
    for spec in (SKINTOKENS, UNIMATE):
        head = ""
        if (spec["dir"] / ".git").exists():
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=spec["dir"], capture_output=True,
                                  text=True).stdout.strip()
        if head != spec["commit"]:
            gaps.append(f"{spec['dir'].name} checkout at {spec['dir']}")
    for f in SKINTOKENS["files"]:
        if not (SKINTOKENS["dir"] / f).exists():
            gaps.append(f"SkinTokens {f}")
    if not (SKINTOKENS["dir"] / "models" / "Qwen3-0.6B" / "config.json").exists():
        gaps.append("Qwen3-0.6B config")
    for f in UNIMATE["files"]:
        if not (UNIMATE["dir"] / "outputs" / f).exists():
            gaps.append(f"UniMate {f}")
    return gaps


def require(log=print) -> None:
    """Fetch whatever ``setup`` would, if any of it is missing."""
    gaps = missing()
    if gaps:
        log(f"[setup] first run: fetching {', '.join(gaps)}")
        setup(log=log)


def require_capture(log=print) -> Path:
    """CoTracker's checkout and weights, fetched on first use. Returns the checkpoint's path."""
    from huggingface_hub import hf_hub_download

    head = ""
    if (COTRACKER["dir"] / ".git").exists():
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=COTRACKER["dir"], capture_output=True,
                              text=True).stdout.strip()
    if head != COTRACKER["commit"]:
        checkout(COTRACKER, log)
    weights = COTRACKER["dir"] / "checkpoints" / COTRACKER["files"][0]
    if weights.exists():
        return weights
    return Path(hf_hub_download(COTRACKER["repo_id"], COTRACKER["files"][0], local_dir=weights.parent))


def setup(weights: bool = True, capture: bool = False, log=print) -> None:
    checkout(SKINTOKENS, log)
    checkout(UNIMATE, log)
    if weights:
        download_weights(log)
    if capture:
        require_capture(log)
