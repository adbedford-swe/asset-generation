"""Generate a 3DGS .ply from an image using Idol, with local-cache encoders.

Bypasses the Hugging Face gated-access check by pointing the two image
encoders (DINOv3, FLUX.2 VAE) at their already-downloaded local snapshot
folders. Only repo *ids* are gated-checked, so local paths load offline.

Usage:
    python pipeline/gen_idol.py --image <img> [--mode synthetic|real]
                                [--out outputs/idol] [--num_gaussians 262144]
                                [--seed 42] [--device cuda:0]
"""
import argparse
import os
import sys
from pathlib import Path

import torch

HF_HUB = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"


def _snapshot(repo_dir_name: str) -> Path:
    """Return the single snapshot folder for a cached HF repo."""
    base = HF_HUB / repo_dir_name / "snapshots"
    snaps = [p for p in base.iterdir() if p.is_dir()]
    if not snaps:
        raise FileNotFoundError(f"No snapshot for {repo_dir_name} under {base}")
    return snaps[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--out", default="outputs/idol")
    ap.add_argument("--num_gaussians", type=int, default=262144)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    dinov3 = _snapshot("models--facebook--dinov3-vith16plus-pretrain-lvd1689m")
    flux2 = _snapshot("models--black-forest-labs--FLUX.2-dev")
    print(f"[gen_idol] dinov3  <- {dinov3}")
    print(f"[gen_idol] flux2   <- {flux2}")

    from idol import IdolPipeline

    pipeline = IdolPipeline.from_pretrained(
        "speridlabs/idol",
        image_encoders={"dinov3": str(dinov3), "flux2_vae": str(flux2)},
        device=args.device,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    output = pipeline(
        args.image,
        mode=args.mode,
        seed=args.seed,
        num_gaussians=args.num_gaussians,
        steps=args.steps,
    )
    stem = Path(args.image).stem
    ply_path = out / f"{stem}.ply"
    output.gaussian.save_ply(str(ply_path))
    output.preprocessed_image.save(str(out / f"{stem}_input.png"))
    print(f"[gen_idol] wrote {ply_path}")
    print(f"[gen_idol] wrote {out / (stem + '_input.png')}")
    print(f"[gen_idol] camera: {output.camera}")


if __name__ == "__main__":
    main()
