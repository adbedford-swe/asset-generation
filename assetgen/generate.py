"""Stage 0: image -> 3D Gaussian splat with Idol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Idol's image encoders are gated on the Hub. When they are already in the local
# cache, pass the snapshot folders: Idol only checks gated access for repo ids.
ENCODERS = {
    "dinov3": ("facebook/dinov3-vith16plus-pretrain-lvd1689m", "config.json"),
    "flux2_vae": ("black-forest-labs/FLUX.2-dev", "vae/config.json"),
}


def cached_snapshot(repo: str, key_file: str) -> str | None:
    """The local snapshot folder holding ``key_file`` of ``repo``, or None if it is not cached."""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo, key_file)
    if not isinstance(path, str):
        return None
    return str(Path(path).parents[len(Path(key_file).parts) - 1])


def resolve_encoders() -> dict:
    return {key: cached_snapshot(repo, f) or repo for key, (repo, f) in ENCODERS.items()}


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
