# asset-generation

One image in, one engine-ready `.glb` out: a textured, rigged character with a set of
text-prompted animation clips. Built for Linux + ROCm (2x RX 7900 XTX), and runs on
CUDA or CPU too.

```
image ──Idol──▶ 3D Gaussian splat ──▶ upright, front-facing, textured mesh
      ──SkinTokens──▶ skeleton + skin weights ──UniMate──▶ motions ──▶ asset.glb (every clip)
```

| Stage | What runs | Output |
|---|---|---|
| `generate` | [Idol](https://github.com/adbedford-swe/idol-sdk) (ROCm fork) | `0_splat/splat.ply`, `input.png`, `camera.json` |
| `mesh` | pure-PyTorch splat rasterizer, TSDF fusion with silhouette carving, marching cubes, quadric decimation, xatlas, occlusion-aware texture bake | `1_mesh/mesh.glb`, `front.png`, `compare.png`, `report.json` |
| `rig` | [SkinTokens / TokenRig](https://github.com/VAST-AI-Research/SkinTokens) | `2_rig/rigged.glb`, `rigged.json`, `preview.png` |
| `animate` | [UniMate](https://github.com/Friedrich-M/UniMate) `rig_preprocess`, sampler and mesh driver | `3_motion/animated/*.glb`, `clips.json` |
| `finalize` | glTF animation merge | `<name>.glb`, with one named animation per prompt |

## Setup (ROCm)

```bash
bash scripts/setup_rocm.sh
```

Idol is installed from the private `adbedford-swe/idol-sdk` repository, so git on the machine needs read
access to it. The script checks first, runs `gh auth setup-git` if `gh` is logged in, and otherwise says what
to fix. It then installs uv if it is missing and builds `.venv` from `uv.lock` (Python 3.11, torch 2.14.1+rocm7.2,
bpy 4.5 as a module, so no Blender install is needed). It then runs `assetgen setup` (pinned SkinTokens and
UniMate checkouts in `third_party/`, plus their weights: about 3 GB) and `assetgen doctor`. Idol's image
encoders (DINOv3, FLUX.2) are gated on Hugging Face: request access and run `hf auth login`, or keep the
copies already in `~/.cache/huggingface`, which are used offline.

`assetgen run` (and the web UI) also fetches the checkouts and weights itself if they are missing, so a run on
a machine where `assetgen setup` never ran or stopped early completes the setup first.

`assetgen doctor` checks the GPU, the fused attention kernels (RDNA3 needs
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, which every stage sets), every import, the checkouts at their
pinned commits, the weights and free disk space.

## Use

```bash
uv run assetgen run --image demon.png --out outputs/demon --height 1.9 \
    --prompt "An object stands idle and breathes." "An object walks forward." \
             "An object swings a weapon overhead." "An object roars."

uv run assetgen batch --images figures/ --out outputs --devices cuda:0,cuda:1   # one asset per GPU at a time
uv run assetgen batch --splats outputs/ --out remeshed --from mesh --devices cuda:0,cuda:1   # <name>/splat.ply (+ input.png)
uv run assetgen run --ply splat.ply --reference input.png --out outputs/x      # start from an existing splat
uv run assetgen run --image demon.png --out outputs/demon --from rig --force    # redo from a stage
```

Or from a browser:

```bash
uv run assetgen webui                    # http://127.0.0.1:7860
uv run assetgen webui --host 0.0.0.0     # reachable from other machines on the network
```

Upload an image (or a splat and its Idol input image), list the motions and press Generate. The log streams,
each stage's preview appears as it finishes (front view, splat against mesh, skeleton, every clip), and the
finished asset plays in an interactive 3D viewer with a clip picker and a download link. Runs land in
`outputs/webui/<time>-<name>/`, laid out like `assetgen run`, so the CLI can resume them.

**Continue a run** picks up an earlier run from any stage: keep a mesh you like and re-rig it, or keep the
rig and generate different motions. Selecting a run shows its results; *Run from this stage* copies its
results up to that stage into a new run folder and runs the rest with the current settings, so the run you
started from is never changed and you can branch it as often as you like.

Each stage runs in its own process, so GPU memory is released between models, and logs to
`<out>/logs/<stage>.log`. A stage is skipped when its outputs exist and it last ran with the same arguments.
Once a stage runs, every later stage runs too.

Prompts follow UniMate's training captions: start with "An object" and describe one motion ("An object
jumps in place.", not "A demon jumps."). Without `--prompt` you get idle, walk and run loops and a one-shot
attack.

**Loops.** Every clip loops seamlessly unless its prompt ends in `[once]` (`"An object attacks forward. [once]"`).
A looping clip's last frame is its first, and the motion through the seam is as smooth as anywhere else in it.
By default loops play in place: a walk's forward travel is removed, and the speed it walked at is stored in the
clip's glTF `extras` (`root_velocity`, m/s, with `loop` and `loop_frames`) so game code can move the character
to match. `--root-motion` keeps the travel instead, for engines that drive movement from the root bone.
`--loop off` turns looping off; `--loop 120` makes 120-frame (4 s) loops, any length up to 600 frames. Finalize measures every seam and
writes it, with the clips' metadata, to `<name>.json`.

Useful options: `--repetitions N` (several samples per prompt), `--cfg-scale` (prompt strength, default 3,
try 5 to 7 when a prompt is ignored), `--rig-class mixamo` (force a Mixamo-template skeleton on a humanoid),
`--num-beams` (SkinTokens beam width, default 10), `--faces` / `--texture` / `--voxels` (mesh budget),
`--llm-base-url http://localhost:8080/v1` (label joints with a local llama.cpp server instead of
UniMate's offline rules), `--mode real` (photographs).

`assetgen synth --mesh Fox.glb --out testdata/fox` turns any textured mesh into an Idol-like splat and
input image (written Y-down, at a known yaw). That's useful for checking the pipeline without Idol.

## How it works, and why

**Orientation.** Idol writes its PLY Y-down (its z-up object turned +90 degrees about X) at whatever yaw
the image implied, and rigging and motion both depend on the asset being upright and facing a known way.
The mesh stage renders the splat around the vertical axis, upright and flipped, and matches each view
against Idol's preprocessed input image (silhouette IoU plus colour agreement, cropped to bounding boxes
so focal length does not matter). It then refines the facing with the splat's mirror-symmetry plane. The
character is taken to face roughly towards the camera of the input image, so use an image that shows its front
or a three-quarter view: one taken from behind gives an asset facing backwards. The
asset comes out Y-up, facing +Z (the glTF convention), centred and standing on y = 0. On synthetic
inputs with known pose, facing is recovered to within 0.2 degrees from three-quarter views.

**Mesh.** The splat is rendered from 64 cameras spread over the sphere (RGB, coverage, median depth),
fused into a truncated signed distance volume where uncovered pixels carve empty space, then meshed and
cleaned: floating islands are dropped, Taubin smoothing is applied, and quadric decimation brings it to
40k faces. The texture is baked by projecting every texel into the views that see it unoccluded (depth
test), weighted towards frontal views. Texels no view sees take the colour of the nearest Gaussians, and
chart borders are dilated. `report.json` scores the mesh against the splat from 8 cameras the bake never
used (colour PSNR, silhouette IoU) and `compare.png` shows the pairs.

**Rig.** SkinTokens runs from its checkout with no Gradio, demo script or Blender server. The mesh goes
through its own `.npz` loader, and the predicted rig is mapped back exactly (its normalisation is a
uniform scale and shift of the same vertices). flash-attn is replaced by an SDPA shim and the Qwen3
backbone uses SDPA, so nothing CUDA-only is needed. Rigs over UniMate's 70-joint budget get their
lightest leaf bones folded into their parents.

**Motion.** Sampling uses a fixed-step solve (32 Euler steps) instead of UniMate's adaptive dopri5 at
atol 1e-6: on the same noise, 50 steps differ from dopri5 by 0.3-1.2% of the features' spread and 25 steps by
0.5-2.4%, at a third to a seventh of the time. All clips are driven in one Blender session, GLB only unless
`--fbx` is given. The rig's UniMate preparation is reused when the rig and its settings have not changed, which
is what continuing a run from the motion stage hits. UniMate's `rig_preprocess` runs without stopping for review. The facing pair it needs comes
from geometry: since the asset faces +Z, it is the most clearly mirrored pair of joints near the root,
with the right-hand one at -X. Joint labels come from UniMate's offline rules (geometric labels for
SkinTokens' unnamed bones) or an LLM. UniMate was written for bpy 4.0. Under bpy 4.5 its rebuilt actions
were left on no action slot (the 4.4 animation system), which exported clips with no animation, so
`assetgen.unimate_compat` assigns the slot. bpy can also crash while the interpreter shuts down after its
files are written; the stage checks for the files instead of trusting that exit code.

**Loops.** UniMate denoises one 60-frame window at a time. `assetgen.motion_sampler` treats each loop as a
ring of L frames, of any length, covered by overlapping 60-frame windows that wrap around its end. Every
sampling step runs all windows of all clips through the model as one batch, blends each frame's velocity over
the windows covering it (weighted towards window centres) and advances the ring; the windows shift every step
so no window edge settles into the motion. This is the MultiDiffusion idea on a circular time axis. A ring has
no ends, so the clip is periodic by construction: there is no seam to blend, and the frames either side of the
wrap are as continuous as any other neighbours. A ring shorter than a window simply wraps within it. All of
UniMate's per-frame features are independent of world position (root-relative joint positions, local rotations,
facing-frame velocities, root height), so equal features mean the same pose moving the same way while the root
still travels. With `--loop auto` (the default) a quick free pass over 120 frames reveals the motion's cycle
(autocorrelation of its standardised joint features, first strong peak) and the loop is one cycle, or the fewest
whole cycles lasting at least 20 frames; a motion without a cycle loops over 3 seconds. On a test fox, 3- and
4-second loops close to within 3e-7 of the asset's size with seams 0.5-0.8x as rough as the rest of the clip,
and the 4-second rings are as smooth inside (median and 99th-percentile frame-to-frame change) as loops made in
a single window, so the windows leave no trace.

**Finalize.** Every clip drives the same canonical asset, so the clips' animations are merged into one
file by node name and named after their prompts. Looping clips are made to loop in place (or keep their root
motion) and get their metadata. `--height` scales the armature root.

## Development

```bash
uv sync --extra dev
uv run pytest
```

The tests cover the rasterizer against a brute-force reference, depth order, axis and quaternion
conventions, sphere reconstruction and texture colours, glTF round trips, orientation recovery, rig
utilities, the facing pair, skinning and the clip merge.

On Windows (development only: CUDA torch from the cu130 index), put the venv and `third_party/` on a short
path (`UV_PROJECT_ENVIRONMENT`, `ASSETGEN_THIRD_PARTY`), since several checkpoint paths pass the 260-character
limit, and set `UV_SYSTEM_CERTS=1` behind a TLS-intercepting proxy. UniMate's preview renderer is skipped
there because it uses `fcntl`.

## Licences

Idol is Apache-2.0 and SkinTokens is MIT. UniMate's code is MIT, but its released checkpoints are
CC BY-NC 4.0, so assets animated with them are for non-commercial use.
