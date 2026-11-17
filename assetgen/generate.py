"""Stage 0: image -> 3D Gaussian splat with Idol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Idol's image encoders are gated on the Hub. When they are already in the local
# cache, pass the snapshot folders: Idol only checks gated access for repo ids.
ENCODERS = {
    "dinov3": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "flux2_vae": "black-forest-labs/FLUX.2-dev",
}


def resolve_encoders() -> dict:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    resolved = {}
    for key, repo in ENCODERS.items():
        try:
            allow = ["vae/*"] if key == "flux2_vae" else None
            resolved[key] = snapshot_download(repo, local_files_only=True, allow_patterns=allow)
        except LocalEntryNotFoundError:
            resolved[key] = repo
    return resolved


def generate(image: str, out_dir: Path, device: str = "cuda", mode: str = "synthetic", seed: int = 42,
             steps: int = 50, num_gaussians: int = 262144, log=print) -> dict:
    from idol import IdolPipeline

    encoders = resolve_encoders()
    for k, v in encoders.items():
        log(f"[generate] {k}: {v}")
    pipeline = IdolPipeline.from_pretrained("speridlabs/idol", image_encoders=encoders, device=device)
    output = pipeline(image, mode=mode, seed=seed, steps=steps, num_gaussians=num_gaussians)
    out_dir.mkdir(parents=True, exist_ok=True)
    output.gaussian.save_ply(str(out_dir / "splat.ply"))
    output.preprocessed_image.save(out_dir / "input.png")
    (out_dir / "camera.json").write_text(json.dumps(output.camera, indent=2))
    log(f"[generate] wrote {out_dir / 'splat.ply'}; camera {output.camera}")
    return output.camera


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--num_gaussians", type=int, default=262144)
    a = ap.parse_args(argv)
    generate(a.image, Path(a.out), a.device, a.mode, a.seed, a.steps, a.num_gaussians)


if __name__ == "__main__":
    main()
