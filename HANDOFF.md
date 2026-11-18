# HANDOFF: Idol → rigged & animated asset pipeline

Read this, then [README.md](README.md). The README says how to run the pipeline and why it is built this way;
this file says what state it is in, what has been checked on which machine, and what to watch.

## Goal

Turn one image into an animated, engine-ready `.glb` with no human in the loop: Idol splat → textured mesh
→ auto-rig for any anatomy (the reference asset is a demon/viking with horns, a tail and an axe, so no
humanoid templates) → text-prompted motion → one file holding the mesh, rig and every clip.

## State

Implemented end to end in the `assetgen` package with one CLI (`assetgen run|batch|setup|doctor|synth`).
It replaces the earlier `pipeline/` scripts entirely; those had these defects:

- The renderer read `rot_0..3` as (x, y, z, w). Idol and the INRIA layout store (w, x, y, z), so every
  Gaussian was oriented wrongly.
- Its depth sort key, `tile_id * 1e6 + depth`, was promoted to float32, so past the first rows of tiles the
  depth part fell below float precision and splats composited in list order.
- Idol's PLY is Y-down (its z-up object turned +90° about X), not Y-up as the previous handoff said, and its
  yaw is arbitrary. Nothing stood the asset up or turned it to face a known direction, which SkinTokens and
  UniMate's facing both depend on.
- Fusion used expected depth (which smears edges) with no silhouette carving. The texture bake had no
  occlusion test and looped over triangles in Python. There was no decimation before xatlas.

## Verified, and where

**On the ROCm machine** (2x RX 7900 XTX): the owner ran the pipeline and the web UI end to end and reported it
working, after reporting and getting fixed the motion stage stalling once Blender had exported (PR #3); that covers
main up to the continue-a-run feature (PR #4). The loop and speed work (PR #5) has not been run there yet.

Everything below ran on the development laptop: **Windows 11, an NVIDIA RTX 500 Ada (4 GB) on CUDA
(torch 2.14.1+cu130), bpy 4.5.16**. The code has no
CUDA-only parts (no compiled extensions; flash-attn replaced by SDPA), and `uv.lock` resolves Linux to
`torch 2.14.1+rocm7.2` and `triton-rocm 3.8.0`. But resolving is not installing, and installing is not
running.

- `uv run pytest`: 20 tests pass (rasterizer vs a brute-force reference, conventions, sphere
  reconstruction, glTF round trips, orientation, rig utilities, skinning, clip merge).
- Stage 1 on Idol's own example splat (`default-obj-backpack.ply` from the Idol HF Space): 64 views at
  768 px, 256³ fusion, 40k faces, 2048² texture in about 90 s on the 4 GB GPU. Colour PSNR against the splat
  is about 25 dB from held-out views.
- Orientation on Idol-like splats made from Khronos' Fox and CesiumMan (`assetgen synth`), written Y-down at
  yaws 20°/137°/250° with input views from −40° to +30°: up flip right every time, facing within 0.2°.
- The whole chain on the Fox (quadruped): mesh → SkinTokens (29 joints: spine, neck, head, ears, four legs,
  tail) → UniMate (facing pair = what UniMate's own rules label Right/Left Thigh) → walk and jump clips →
  merged `fox.glb` with named clips, checked by evaluating the skinning per frame (`assetgen.gltf_anim`).
- `assetgen run` end to end on CesiumMan (humanoid) through the orchestrator, including a resume after an
  interrupted run (finished stages reported up to date). Result: 22-joint humanoid skeleton (hips, spine,
  neck, head, clavicles, arms, legs, toes), `walks_forward` and `waves_with_its_right` clips (the raised arm
  is the character's right), scaled to 1.8 m. Stage times on the 4 GB laptop GPU: mesh about 100 s after the
  profiling fixes (645 s before them), rig 120 s (2 beams), motion 325 s for two clips, finalize 13 s. Most
  of the motion time is bpy importing and exporting each clip, which is CPU-bound and costs the same on any GPU.

- Linux (Ubuntu 20.04 under WSL2 on the laptop, no GPU): `setup_rocm.sh` resolved and built the whole lock,
  including the private Idol package (GitHub credentials needed). ROCm torch 2.14.1 (HIP 7.2) installed: the
  6.2 GB wheel was fetched separately because the office proxy kept dropping it, and its hash matched `uv.lock`.
  `assetgen setup` fetched both checkouts and all weights, and 27/27 tests passed. This turned up and fixed:
  open3d 0.20 needs glibc 2.35 (now pinned below 0.20), and bpy needs X11/GL libraries even headless
  (`setup_rocm.sh` now checks and prints the packages to install).
- A full `assetgen run` under Linux on CPU (the Fox, through mesh, rig, motion and finalize) ending
  in an animated `fox.glb`. It surfaced three CPU-only gaps, now fixed: SkinTokens probing the GPU name at import,
  SkinTokens needing CPU autocast, and UniMate's config sampling on `cuda`. It also confirmed the bpy teardown fix:
  Blender as a module reported leaked memory and then crashed (Windows) or hung (Linux, which stalled the web UI
  before its clips appeared) at interpreter exit. Blender steps now end without that teardown.
- `batch` with a GPU worker and a CPU worker running at once, and the web UI driven in a browser through
  every stage.

Not verified anywhere: the `generate` stage against real Idol (Idol's gated encoders aren't on the laptop; its
calls are checked against Idol's API and tested with a stub), two real GPUs at once, any AMD GPU kernel, and
`--llm-base-url` labelling.

## Loops and motion speed

Clips loop seamlessly by default (prompts ending in `[once]` do not); see "Loops" in the README for the method.
The sampler (`assetgen.motion_sampler`) replaces UniMate's sampling through `assetgen.unimate_compat`, which
patches `unimate.inference.generate.generate_samples` before `sample.py` imports it; in-betweening, editing and
expansion still go to UniMate's own code. Measured on the laptop (Fox): looped clips close to within 1e-7 of the
asset size with seam roughness 0.1-0.8x the rest of the clip; a one-shot clip misses by 0.11 of its size.
Loops play in place unless `--root-motion`; each clip's glTF `extras` carry `loop`, `loop_frames`,
`root_motion` and `root_velocity` (m/s). Continuing a run from Motion reuses its UniMate rig preparation and the
motion stage drives every clip in one Blender session (GLB only unless `--fbx`): a two-clip continuation took
about 130-165 s on the laptop against about 265 s before. Still open: the 60-frame window caps a loop at 48 frames
(1.6 s); longer cycles would need UniMate's motion expansion with the same tie across segments.

## First thing to do on the ROCm box

```bash
bash scripts/setup_rocm.sh          # env, checkouts, weights, then doctor
uv run assetgen run --image <idol-sdk>/demo/examples/fig-viking.webp --out outputs/viking --height 1.9
```

Check `outputs/viking/1_mesh/front.png` (the asset should face you, upright), `1_mesh/compare.png`,
`2_rig/preview.png`, then the final `viking.glb` in a viewer. Things to watch:

- `doctor` reporting no fused attention: SDPA then falls back to the math kernel, which is much slower.
- The iGPU appearing as a torch device: pin the 7900 XTXs with `HIP_VISIBLE_DEVICES=0,1` if so.
- Disk: `setup_rocm.sh` keeps the uv cache next to the repo so the 6.2 GB torch wheel is hard-linked into
  `.venv`. The env is roughly 15–20 GB, plus about 4 GB of weights in `third_party/` and the HF cache.
- llama.cpp: the earlier session freed VRAM with `suspend_llama_cpp` for GPU commands. The pipeline needs at
  most about 14 GB on one card (SkinTokens at 10 beams), so on a 24 GB card that should not be needed.

## Machine facts carried over

- 2× RX 7900 XTX (gfx1100, 24 GB) + iGPU; ROCm drives them as `cuda:0` / `cuda:1`.
- System Python is 3.14 and externally managed. The pipeline uses its own uv-managed 3.11 venv (bpy 4.5
  only ships cp311 wheels and pins numpy < 2), so system packages are untouched.
- Idol weights and its gated encoders are cached in `~/.cache/huggingface/hub`. `assetgen.generate` resolves
  the encoders to those local snapshots so no token is needed.
- Blender 5.3 is installed, but the pipeline does not use it (bpy 4.5 is in the venv). Blender 5 removed
  the legacy action API that UniMate's code relies on.

## Possible improvements

- Score several SkinTokens samples (`--seed`) and keep the best: joints inside the mesh, symmetry, joint
  count.
- Loopable idle/walk clips: UniMate's in-betweening (`--keep_frames 0,-1`) can pin the last frame to the
  first.
- Longer clips with UniMate's motion expansion (chained prompts).
