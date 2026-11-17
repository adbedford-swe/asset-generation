# HANDOFF — Idol → Rigged & Animated Asset Pipeline

> Read this top-to-bottom before doing anything. It captures the goal, the
> research, what is already built and validated, what is next, and every
> environment gotcha that cost time to discover.

## 1. The overarching task

Turn `speridlabs/idol`'s raw 3D Gaussian Splat (`.ply`) output into an
**animated, engine-ready `.glb`** in a fully automated Python/CLI pipeline.
Four stages (from the original LLM plan, adapted to this ROCm machine):

1. **Stage 1 — `.ply` → textured watertight `.glb`** (mesh extraction + UV + texture).
2. **Stage 2 — topology-agnostic auto-rigging & skinning** (skeleton + LBS weights).
3. **Stage 3 — text-conditioned any-skeleton motion generation**.
4. **Orchestration glue** — one orchestrator passing file paths
   (`asset.ply → mesh.glb → rigged.glb → animated.glb`), plus a headless-Blender
   finalizer (prune floating islands, normalize scale/origin, bake rotations onto
   the armature, export a single self-contained `.glb`).

The user's reference asset is a demon/viking with an axe, horns, and a tail —
i.e. **non-standard anatomy**, which is exactly why template humanoids (Mixamo)
are out and we need arbitrary-skeleton rigging + motion.

## 2. This machine (critical facts)

- **GPU**: 2× AMD Radeon RX 7900 XTX (RDNA3, **gfx1100**, 24 GB each) + 1 iGPU.
  Driven through ROCm as `cuda` / `cuda:0` / `cuda:1`. torch is
  **2.12.0+rocm7.2** (system python3.14). `torch.cuda.is_available()` → True.
- **Python**: only **3.14** is installed system-wide. `uv` is at
  `/home/alex/.local/bin/uv` (use it to make venvs with other Python versions —
  several downstream tools need 3.10–3.13).
- **idol**: installed in `~/.local/lib/python3.14/site-packages/idol` (NOT
  editable). Source repo is at `idol-sdk/` in this workspace. Weights are cached
  in `~/.cache/huggingface/hub` (idol, dinov3-vith16plus, FLUX.2-dev all present).
- **Blender 5.3.0**: `/home/alex/Documents/Blender/blender-5.3.0-linux-x64/blender`.
  Headless works (`blender -b -P script.py`). **Native 3DGS `.ply` import + render
  works, but exporting a mesh FROM a splat is NOT supported** (user-confirmed) —
  so Blender is only a possible *renderer*, never the mesh extractor.
- **llama.cpp** (`llama-server`) runs on the iGPU and is the local LLM. For any
  GPU-heavy command, pass `suspend_llama_cpp: true` to `execute_command` so it
  stops/restarts the server and frees VRAM.
- **Disk**: root `/` was **100% full** — I purged the pip cache (freed ~21 GB).
  The Storage drive (`/run/media/alex/Storage`, where this workspace lives) has
  ~73 GB free. **Watch disk space**; big model downloads go here or to `~/.cache`.
- **PEP 668**: system python is externally managed. Install with
  `pip install --user --break-system-packages <pkg>` (lands in `~/.local`, same as idol).
- **Browser tool is broken** on this box (Chromium sandbox mount-namespace error).
  Do web research with `curl` against raw.githubusercontent.com / api.github.com,
  or the `web_search` tool.

### torch 2.12 gotchas (already hit)
- `F.normalize(x)` with `dim=None` **crashes on 1-D inputs** ("Dimension out of
  range"). Always pass `dim=-1`.
- `torch.cross(a, b)` without `dim` prints a deprecation warning (harmless).
- `torch.bmm` needs both operands 3-D; use `torch.einsum` for `(3,3) @ (N,3,3)`.

## 3. Research findings (all verified REAL, not hallucinated)

| Stage | Primary choice | Repo / source | ROCm status on THIS box |
|---|---|---|---|
| 1 render | **pure-PyTorch splat rasterizer** (custom) | — (we wrote it) | ✅ works (validated) |
| 1 render (alt) | gsplat | nerfstudio-project/gsplat | ❌ CUDA-only; the HIP port (AMD-Ecosystem/gsplat-amd) targets **Instinct** (gfx942), consumer Radeon = build friction. Dropped. |
| 1 mesh | **Open3D ScalableTSDFVolume** (CPU) | open3d 0.20.0 | ✅ installed |
| 1 UV | **xatlas** | xatlas 0.0.11 | ✅ installed |
| 1 export | **trimesh** | trimesh 5.2.1 | ✅ installed |
| 2 rig | **SkinTokens / TokenRig** (VAST-AI, successor to UniRig) | VAST-AI-Research/SkinTokens | ⚠️ needs flash-attn (patch to SDPA) + bpy module (bpy 5.3.0 has a **cp313** wheel → use Python 3.13 via uv). 14 GB VRAM OK. |
| 2 rig (alt) | MagicArticulate (CVPR 2025) | Seed3D/MagicArticulate | ⚠️ also needs flash-attn; released code is skeleton-gen focused. |
| 2 rig (alt) | UniRig (SIGGRAPH '25) | VAST-AI-Research/UniRig | ❌ needs spconv (CUDA) — bad on ROCm. |
| 3 motion | **UniMate** (SIGGRAPH Asia 2026) | Friedrich-M/UniMate | ⚠️ swap the `torch==2.5.1+cu124` pin for the ROCm build; torch-geometric / torchdiffeq / Motion / CLIP / T5 are pure-Python or have CPU paths. 74 M params, small. |
| 3 motion (alt) | SAMoR (NeurIPS 2026) | arXiv 2607.02148 | research paper, less clear CLI. |

### Key API facts learned
- **idol**: `IdolPipeline.from_pretrained("speridlabs/idol", image_encoders={...}, device=...)`
  then `pipeline(image, mode=, seed=, num_gaussians=, steps=)`. `device` goes to
  `from_pretrained`, **not** `__call__`. Passing **local folder paths** to
  `image_encoders` bypasses the HF gated-access check (only repo *ids* are gated).
  Output `.ply` is standard 3DGS: `x,y,z, nx,ny,nz, f_dc_0..2, opacity,
  scale_0..2, rot_0..3` where **rot_0..3 = (x,y,z,w)** (INRIA convention).
- **SkinTokens**: `python demo.py --input mesh.glb --output rigged.glb [--use_transfer]`.
  flash-attn is imported in `tokenrig.py` + `skin_vae_model.py` with a fallback that
  STILL needs the `flash_attn` package — **patch those two to SDPA** for ROCm.
  `attention_processor.py`, `miche_transformer_blocks.py`, and
  `michelangelo/.../transformer_blocks.py` already fall back to SDPA. Uses a
  `bpy_server.py` subprocess (needs the `bpy` pip module) for mesh transfer/export.
- **UniMate**: `python -m unimate.inference.sample --exp_dir
  outputs/unimate_uniml3d_f60_v3 --asset <asset_dir> --prompt "An object ..."
  --output_dir ...`. Checkpoints: `hf download Linzhan/UniMate --include
  "unimate_uniml3d_f60_v3/*"`. Custom rigged assets go through
  `data_process/rig_preprocess/` which **labels joints with an LLM and pauses for a
  human review** of labels + facing — this is the main HITL step to automate or
  script around. Prompts must start with "An object" and describe one motion.
  Supports motion **expansion** (`--expand_overlap`, chain prompts),
  **in-betweening** (`--keep_frames`), and **editing** (`--keep_joints`).
  `bash scripts/run_animate_motion.sh <dir>` drives each asset's mesh → animated GLB+FBX.

## 4. What is already BUILT and VALIDATED

All under `pipeline/` in this workspace.

- **`pipeline/gen_idol.py`** — generates a `.ply` from an image using idol with
  local-cache encoders (bypasses gated check). **Validated**: produced
  `outputs/idol/fig-viking.ply` (262,144 Gaussians, Y-up, centered at origin,
  ~1 unit tall). Usage:
  `python3 pipeline/gen_idol.py --image <img> --mode synthetic --out outputs/idol --device cuda:0`
  (run with `suspend_llama_cpp: true`).
- **`pipeline/splat_render.py`** — **pure-PyTorch 3DGS forward renderer** (the
  core Stage-1 renderer; no C++/HIP, no gsplat, no Blender). Loads the `.ply`,
  builds 3D covariances from scale+quat, projects to 2D covariances via the
  projection Jacobian, PSD-clamps with `torch.linalg.eigh`, computes the conic +
  3σ radius, does vectorized tile assignment, and front-to-back alpha-composites
  per 16×16 tile. Emits **RGB + expected-depth + coverage + `poses.json`** (view
  matrices are world→camera, camera looks down +Z, X right, Y down).
  **Validated on CPU AND ROCm GPU — pixel-identical output**, a clean viking
  (horned helmet, shield, axe, green tunic, fur, leather belt). ~0.57 s/view on GPU.
  Key functions: `load_gaussians`, `orbit_cameras`, `render_view`, `render_views`.
- **`pipeline/mesh_extract.py`** — full Stage 1: render N orbit views →
  **Open3D ScalableTSDFVolume** fusion (marching cubes, cleans non-manifold /
  duplicated / degenerate, orients normals) → **xatlas** UV unwrap →
  **vectorized multi-view visibility-weighted texture bake** (the only Python loop
  is over triangles; barycentric rasterization, projection into all views, RGB
  gather, and visibility are batched numpy) → **trimesh `.glb`** export with the
  baked albedo. **Written but NOT yet run end-to-end** — that is the immediate next step.
  Usage:
  `python3 pipeline/mesh_extract.py --ply outputs/idol/fig-viking.ply --out outputs/mesh --n_views 32 --image_size 512 --tex_size 1024 --device cuda:0`
- **`pipeline/test_render.py`** — renderer smoke test (renders a few views).

## 5. IMMEDIATE NEXT STEP (do this first)

Run Stage 1 end-to-end and **inspect the result**:
```
python3 pipeline/mesh_extract.py --ply outputs/idol/fig-viking.ply \
    --out outputs/mesh --n_views 32 --image_size 512 --tex_size 1024 --device cuda:0
```
(run with `suspend_llama_cpp: true`). Then verify `outputs/mesh/mesh.glb`:
- Load it (trimesh, or open in Blender headless and render a turntable) and LOOK at
  it. Check: is the mesh watertight and the right shape? Are the UVs sane? Does the
  baked texture match the viking's colors?
- Likely issues to watch: TSDF voxel size vs object scale (object is ~1 m; defaults
  are `voxel_size=0.005`, `sdf_trunc=0.02` — tune if the mesh is too coarse/noisy);
  texture-bake speed at `tex_size=1024` (drop to 512 to iterate faster); thin
  geometry (axe handle, horns) may need more views or a finer voxel.
- If the texture bake is too slow or ugly, a robust fallback is **vertex colors**
  (assign each vertex the color from the best-seeing view) exported without a UV
  texture — still a valid "textured mesh" for the rigger.

## 6. Then: Stage 2 (rigging) and Stage 3 (motion) — NOT started

**Stage 2 (SkinTokens)** plan:
1. `uv venv --python 3.13` (bpy 5.3.0 has a cp313 wheel). Install SkinTokens
   deps + the ROCm torch build + `bpy==5.3.0` (from download.blender.org/pypi).
2. Patch `src/model/tokenrig.py` and `src/model/skin_vae_model.py` flash-attn
   fallbacks to SDPA (mirror the existing SDPA fallback in `attention_processor.py`).
3. `python download.py --model` to fetch TokenRig + FSQ-CVAE checkpoints.
4. `python demo.py --input outputs/mesh/mesh.glb --output outputs/rig/rigged.glb --use_transfer`.
   Verify the skeleton + skin weights look right (open in Blender).
- Fallback if SkinTokens is too much friction on ROCm: **MagicArticulate**.

**Stage 3 (UniMate)** plan:
1. `uv venv --python 3.10` (UniMate pins bpy==4.0.0 cp310; but export/animate can
   run via `blender -b -P` instead of the bpy module, so a newer python may work —
   test it). Install requirements with the ROCm torch swap.
2. `hf download Linzhan/UniMate --include "unimate_uniml3d_f60_v3/*"`.
3. Run `data_process/rig_preprocess/` on `outputs/rig/rigged.glb` to build an asset
   dir (this is the LLM-joint-labeling + review step — script the review or auto-approve).
4. `python -m unimate.inference.sample --exp_dir ... --asset <asset_dir> --prompt
   "An object swings a heavy weapon overhead." --output_dir ...` then
   `bash scripts/run_animate_motion.sh <dir>` to drive the mesh → animated GLB.

**Orchestration**: write a top-level `pipeline/run.py` (or shell script) that chains
gen_idol → mesh_extract → rig → animate, passing file paths, with each stage
skippable/resumable. Add the headless-Blender finalizer at the end.

## 7. Working style notes (for the next session)

- The user is **sensitive to overconfident assertions from partial evidence**
  (e.g. grepping one file and declaring a feature absent). Verify thoroughly —
  read the repo, the docs, the build system — before asserting. When unsure, say so.
- The user actively co-designs and will propose optimizations (e.g. the pure-PyTorch
  renderer). Engage with the design, don't just execute.
- Prefer doing the work over more research ping-pong once a decision is made.
- Use `suspend_llama_cpp: true` for GPU commands. Watch disk space. Use
  `pip install --user --break-system-packages`.

## 8. File map (this workspace)

```
asset-generation/
├── HANDOFF.md                  ← this file
├── idol-sdk/                   ← idol source (installed separately in ~/.local)
│   └── demo/examples/          ← input images (fig-viking.webp, fig-knight.webp, ...)
├── pipeline/
│   ├── gen_idol.py             ← image → .ply  (VALIDATED)
│   ├── splat_render.py         ← pure-PyTorch 3DGS renderer  (VALIDATED)
│   ├── mesh_extract.py         ← Stage 1: .ply → textured .glb  (written, not yet run)
│   └── test_render.py          ← renderer smoke test
└── outputs/
    ├── idol/fig-viking.ply     ← generated test asset (262,144 Gaussians)
    ├── idol/fig-viking_input.png
    └── render_test/            ← renderer test outputs (v000.png, *_depth.png, poses.json)
```
