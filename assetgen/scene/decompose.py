"""Take a picture of a whole scene apart into objects, and work out where each one stands.

One picture of an environment (concept art, a generated view, a photo) already has
what a set of separately generated props lacks: one look, and an arrangement. This
turns it into a scene spec:

1. find the objects named (Grounding DINO, open-vocabulary detection);
2. cut each one out (SAM, prompted with its box), as a picture with transparency;
3. estimate depth (Depth Anything V2, metric), back-project the picture into 3D, and
   fit the ground plane to what is not an object (RANSAC);
4. stand each object on that plane: where its base meets the ground, how tall it is,
   and from which direction the picture saw it;
5. look straight down on the plane: the picture's own ground, resampled onto the
   scene's ground with the objects taken out, becomes the ground texture, so paths and
   patches lie where the picture has them.

Each kind of object becomes one entry in the spec, made from its clearest cut-out
(the largest one least hidden by others) and placed wherever the picture has one.
Its turn is given as ``{"seen_from": <yaw>}``: once the object is made, the mesh
stage knows where Idol's camera was relative to its front, so layout turns it to
show the scene's camera the side the picture showed.

The geometry (:func:`locate`) is plain numpy and runs on any masks and depth map;
the models are only needed by :func:`analyse`.

    assetgen scene decompose concept.png --objects "tent, campfire, pine tree, log" --out camp.json
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DETECTOR = "IDEA-Research/grounding-dino-tiny"
SEGMENTER = "facebook/sam-vit-base"
# A detection whose points form a thin sheet lying along the ground is a patch of it (a field, a path), not an
# object: its plane is this close to the ground's (cosine of the angle) and this thin for its spread.
SHEET_LEVEL, SHEET_THIN = 0.95, 0.15
DEPTH = {"outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
         "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"}


@dataclass(eq=False)
class Found:
    label: str
    score: float
    box: list[float]                 # x0, y0, x1, y1 in pixels
    mask: np.ndarray | None = None   # (H, W) bool


@dataclass(eq=False)
class Placed:
    label: str
    position: list[float]            # x, z on the ground, metres
    height: float                    # metres
    width: float
    seen_from: float                 # yaw of the direction from the object to the camera, degrees
    visible: float                   # share of its outline not hidden by other objects or the picture's edge
    area: int                        # mask pixels
    found: Found = field(repr=False, default=None)
    sheet: bool = False              # its points lie in a thin layer along the ground: a patch of ground


@dataclass
class Ground:
    normal: np.ndarray               # unit, camera coordinates, pointing up (towards the camera's side)
    offset: float                    # plane: normal . p + offset = 0
    origin: np.ndarray               # world origin in camera coordinates (on the plane)
    axes: np.ndarray                 # rows: world x, y, z in camera coordinates
    camera_height: float

    def to_world(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points) - self.origin) @ self.axes.T


def intrinsics(width: int, height: int, fov_deg: float) -> tuple[float, float, float]:
    """Focal length and principal point for a horizontal field of view."""
    f = 0.5 * width / math.tan(math.radians(fov_deg) / 2)
    return f, width / 2, height / 2


def back_project(depth: np.ndarray, fov_deg: float) -> np.ndarray:
    """(H, W, 3) camera-space points (x right, y down, z forward) from a depth map along the view axis."""
    H, W = depth.shape
    f, cx, cy = intrinsics(W, H, fov_deg)
    v, u = np.mgrid[0:H, 0:W].astype(np.float64) + 0.5
    return np.stack([(u - cx) / f * depth, (v - cy) / f * depth, depth], axis=-1)


def fit_ground(points: np.ndarray, iterations: int = 400, seed: int = 0) -> tuple[np.ndarray, float]:
    """RANSAC plane through ``points`` (N, 3): unit normal facing the camera's up (-y), and offset."""
    rng = np.random.default_rng(seed)
    pts = points[np.isfinite(points).all(1)]
    if len(pts) < 50:
        raise ValueError("too little ground is visible to find the ground plane")
    if len(pts) > 40000:
        pts = pts[rng.choice(len(pts), 40000, replace=False)]
    tol = 0.01 * float(np.median(pts[:, 2]))
    best, best_count = None, -1
    for _ in range(iterations):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        if n[1] > 0:
            n = -n
        if -n[1] < 0.3:  # the ground cannot stand steeper than ~70 degrees to the view
            continue
        d = -float(n @ a)
        count = int((np.abs(pts @ n + d) < tol).sum())
        if count > best_count:
            best, best_count = (n, d), count
    if best is None:
        raise ValueError("found no ground plane in the picture")
    n, d = best
    inliers = pts[np.abs(pts @ n + d) < tol]
    centre = inliers.mean(0)
    _, _, vt = np.linalg.svd(inliers - centre, full_matrices=False)
    n = vt[2] if vt[2][1] < 0 else -vt[2]
    return n, -float(n @ centre)


def ground_frame(normal: np.ndarray, offset: float, anchor: np.ndarray) -> Ground:
    """World axes on the plane: y up, z towards the camera, x to the camera's right; origin below ``anchor``."""
    n = normal / np.linalg.norm(normal)
    origin = anchor - (n @ anchor + offset) * n
    towards_camera = -origin - (n @ -origin) * n  # the camera sits at the camera-space origin
    if np.linalg.norm(towards_camera) < 1e-9:      # looking straight down: take the picture's down as +z
        towards_camera = np.array([0.0, 1.0, 0.0]) - n[1] * n
    z = towards_camera / np.linalg.norm(towards_camera)
    x = np.cross(n, z)
    return Ground(n, offset, origin, np.stack([x, n, z]), camera_height=float(abs(offset)))


def _is_sheet(points: np.ndarray) -> bool:
    """Whether world-space points lie in a thin layer parallel to the ground."""
    if len(points) < 50:
        return False
    sample = points[:: max(1, len(points) // 20000)]
    _, spread, axes = np.linalg.svd(sample - sample.mean(0), full_matrices=False)
    return abs(axes[2][1]) > SHEET_LEVEL and spread[2] < SHEET_THIN * spread[1]


def _visible_share(mask: np.ndarray, others: np.ndarray) -> float:
    """How much of a mask's outline borders open background rather than other objects or the picture's edge."""
    from scipy.ndimage import binary_dilation

    H, W = mask.shape
    ring = binary_dilation(mask, iterations=2) & ~mask
    if not ring.any():
        return 0.0
    edge = np.zeros_like(mask)
    edge[:3, :] = edge[-3:, :] = edge[:, :3] = edge[:, -3:] = True
    blocked = ring & (others | edge)
    touching_edge = (mask & edge).any()
    return float(1 - blocked.sum() / ring.sum()) * (0.5 if touching_edge else 1.0)


def locate(found: list[Found], depth: np.ndarray, fov_deg: float = 55.0) -> tuple[list[Placed], Ground]:
    """Stand every found object (with its mask) on the ground plane that the rest of the picture shows."""
    from scipy.ndimage import binary_dilation

    points = back_project(depth, fov_deg)
    H, W = depth.shape
    objects = np.zeros((H, W), bool)
    for f in found:
        objects |= f.mask
    open_ground = ~binary_dilation(objects, iterations=4)
    open_ground[: H // 4] = False  # the top of a picture is mostly far away, sky or backdrop
    normal, offset = fit_ground(points[open_ground])
    anchors = []
    for f in found:
        p = points[f.mask]
        heights = -(p @ normal + offset)
        anchors.append(p[heights <= np.percentile(heights, 10)].mean(0))
    ground = ground_frame(normal, offset, np.mean(anchors, axis=0) if anchors else points[H // 2, W // 2])
    camera = ground.to_world(np.zeros(3))
    placed = []
    for f, anchor in zip(found, anchors):
        p = ground.to_world(points[f.mask])
        h = p[:, 1]
        top = float(np.percentile(h, 98))
        base = p[h <= np.percentile(h, 10)]
        x, z = float(np.median(base[:, 0])), float(np.median(base[:, 2]))
        across = np.percentile(p[:, 0], 98) - np.percentile(p[:, 0], 2)
        others = objects & ~f.mask
        placed.append(Placed(f.label, [round(x, 3), round(z, 3)], round(max(top, 1e-3), 3), round(float(across), 3),
                             round(math.degrees(math.atan2(camera[0] - x, camera[2] - z)) % 360, 2),
                             round(_visible_share(f.mask, others), 3), int(f.mask.sum()), f, _is_sheet(p)))
    return placed, ground


def ground_texture(image: np.ndarray, objects: np.ndarray, ground: Ground, fov_deg: float, size,
                   pixels: int = 1024) -> tuple[np.ndarray, float]:
    """The picture's ground seen from straight above, over ``size`` = (width, depth) metres centred on the
    scene's origin: rows run along +z, columns along +x, as the ground's UVs do. Where the picture does not
    show the ground (behind objects, outside the frame) the texture is filled from the ground around it.
    Returns the texture and the share of it the picture showed."""
    from scipy.ndimage import binary_dilation, map_coordinates

    H, W = image.shape[:2]
    f, cx, cy = intrinsics(W, H, fov_deg)
    width, depth = float(size[0]), float(size[1])
    cols = pixels if width >= depth else max(16, int(round(pixels * width / depth)))
    rows = pixels if depth >= width else max(16, int(round(pixels * depth / width)))
    xs = (np.arange(cols) + 0.5) / cols * width - width / 2
    zs = (np.arange(rows) + 0.5) / rows * depth - depth / 2
    X, Z = np.meshgrid(xs, zs)
    world = np.stack([X, np.zeros_like(X), Z], axis=-1)
    cam = ground.origin + world @ ground.axes
    ahead = cam[..., 2] > 1e-6
    zc = np.where(ahead, cam[..., 2], 1.0)
    u = f * cam[..., 0] / zc + cx - 0.5
    v = f * cam[..., 1] / zc + cy - 0.5
    hidden = binary_dilation(objects, iterations=max(2, W // 200))
    inside = ahead & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    shown = inside.copy()
    shown[inside] = ~hidden[np.round(v[inside]).astype(int), np.round(u[inside]).astype(int)]
    texture = np.zeros((rows, cols, 3))
    for c in range(3):
        texture[..., c] = map_coordinates(image[..., c].astype(np.float64), [v, u], order=1, mode="nearest")
    if not shown.any():
        return np.full((rows, cols, 3), image[..., :3].reshape(-1, 3).mean(0), np.uint8), 0.0
    out = np.where(shown[..., None], texture, fill(texture, shown))
    return np.clip(out, 0, 255).astype(np.uint8), float(shown.mean())


def fill(image: np.ndarray, known: np.ndarray) -> np.ndarray:
    """``image`` with its unknown pixels filled from the known ones around them (push-pull: average the known
    pixels down a pyramid, then blend back up, so every gap takes the colour of its surroundings at its own scale)."""
    from scipy.ndimage import zoom

    levels = []
    colour, weight = image * known[..., None], known.astype(np.float64)
    while min(weight.shape) > 1:
        levels.append((colour, weight))
        h, w = weight.shape
        colour = np.pad(colour, ((0, h % 2), (0, w % 2), (0, 0)))
        weight = np.pad(weight, ((0, h % 2), (0, w % 2)))
        colour = colour[0::2, 0::2] + colour[1::2, 0::2] + colour[0::2, 1::2] + colour[1::2, 1::2]
        weight = weight[0::2, 0::2] + weight[1::2, 0::2] + weight[0::2, 1::2] + weight[1::2, 1::2]
    result = colour / np.maximum(weight, 1e-9)[..., None]
    for colour, weight in reversed(levels):
        h, w = weight.shape
        coarse = zoom(result, (h / result.shape[0], w / result.shape[1], 1), order=1, mode="nearest")[:h, :w]
        own = colour / np.maximum(weight, 1e-9)[..., None]
        alpha = np.clip(weight, 0, 1)[..., None]
        result = own * alpha + coarse * (1 - alpha)
    return result


def distinct(found: list[Found], inside: float = 0.7) -> tuple[list[Found], list[Found]]:
    """Keep one detection per object: one whose mask lies mostly inside a likelier one's (a cart found on a barn's
    wall) is dropped. Returns (kept, dropped)."""
    kept, dropped = [], []
    for f in sorted(found, key=lambda f: -f.score):
        area = max(int(f.mask.sum()), 1)
        if any((f.mask & k.mask).sum() > inside * area for k in kept):
            dropped.append(f)
        else:
            kept.append(f)
    return [f for f in found if f in kept], dropped


def cutout(image: np.ndarray, mask: np.ndarray, margin: float = 0.12):
    """The object on its own: a square RGBA picture around the mask, transparent elsewhere."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    alpha = (gaussian_filter(mask.astype(np.float32), 0.7) * 255).clip(0, 255).astype(np.uint8)
    rgba = np.concatenate([image[..., :3], alpha[..., None]], axis=-1)
    canvas = np.zeros((side, side, 4), np.uint8)
    top, left = cy - side // 2, cx - side // 2
    sy0, sx0 = max(top, 0), max(left, 0)
    sy1, sx1 = min(top + side, mask.shape[0]), min(left + side, mask.shape[1])
    canvas[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = rgba[sy0:sy1, sx0:sx1]
    return Image.fromarray(canvas, "RGBA")


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "object"


def to_spec(placed: list[Placed], image: np.ndarray, parts: Path, name: str, style: str = "",
            each: bool = False, redraw: float = 0.0, scale: float = 1.0, concept: str | None = None,
            prompts: dict[str, str] | None = None, ground: Ground | None = None, fov_deg: float = 55.0) -> dict:
    """The scene spec for ``placed``; cut-outs are written to ``parts``. Objects of one label share one prop
    (made from the clearest cut-out) unless ``each``. ``scale`` multiplies every distance and height."""
    parts = Path(parts)
    parts.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Placed]] = {}
    for p in placed:
        key = f"{slug(p.label)}_{len(groups)}" if each else slug(p.label)
        groups.setdefault(key, []).append(p)
    objects, reach = [], 1.0
    for key, members in groups.items():
        best = max(members, key=lambda p: p.visible * p.area)
        cutout(image, best.found.mask).save(parts / f"{key}.png")
        height = float(np.median([p.height for p in members])) * scale
        place = []
        for p in members:
            x, z = p.position[0] * scale, p.position[1] * scale
            reach = max(reach, abs(x) + p.width * scale, abs(z) + p.width * scale)
            entry = {"position": [round(x, 3), round(z, 3)], "yaw": {"seen_from": p.seen_from}}
            if abs(p.height * scale / height - 1) > 0.05:
                entry["scale"] = round(p.height * scale / height, 3)
            place.append(entry)
        obj = {"id": key, "image": f"{parts.name}/{key}.png", "height": round(height, 3), "place": place}
        label = best.label
        obj["prompt"] = (prompts or {}).get(label, label)
        if redraw > 0:
            obj["redraw"] = redraw
        objects.append(obj)
    size = math.ceil(2 * reach + 2)
    spec = {"name": name, "style": style, "ground": {"size": [size, size]}, "objects": objects}
    if ground is not None:
        from PIL import Image

        objects_mask = np.zeros(image.shape[:2], bool)
        for p in placed:
            objects_mask |= p.found.mask
        texture, shown = ground_texture(image, objects_mask, ground, fov_deg, (size / scale, size / scale))
        Image.fromarray(texture).save(parts / "ground.png")
        spec["ground"].update(texture=f"{parts.name}/ground.png", tile=size)
    if concept:
        spec["concept"] = concept
    return spec


def detect(image, names: list[str], box_threshold: float = 0.25, text_threshold: float = 0.2,
           device: str = "cpu") -> list[Found]:
    """Every object named in ``names`` (Grounding DINO), one box each after removing duplicates."""
    import torch
    from torchvision.ops import nms
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(DETECTOR)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).to(device).eval()
    text = ". ".join(n.strip().lower() for n in names) + "."
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[image.size[::-1]])[0]
    boxes, scores = result["boxes"].cpu(), result["scores"].cpu()
    labels = [str(lbl) for lbl in result.get("text_labels", result.get("labels"))]
    W, H = image.size
    found = []
    for label in sorted(set(labels)):
        idx = [i for i, lbl in enumerate(labels) if lbl == label]
        keep = nms(boxes[idx], scores[idx], 0.5).tolist()
        for k in keep:
            x0, y0, x1, y1 = boxes[idx[k]].tolist()
            if (x1 - x0) * (y1 - y0) > 0.6 * W * H or not label:
                continue  # a box over most of the picture is the ground or the whole scene, not an object
            found.append(Found(_closest(label, names), float(scores[idx[k]]), [x0, y0, x1, y1]))
    return found


def _closest(label: str, names: list[str]) -> str:
    """The name a detected phrase belongs to (the detector may return part of a name, or two run together)."""
    words = set(label.split())
    return max(names, key=lambda n: (len(words & set(n.lower().split())), -abs(len(n) - len(label))))


def segment(image, found: list[Found], device: str = "cpu") -> None:
    """Give every found object its mask (SAM, prompted with its box); overlaps go to the nearer-scoring mask."""
    import torch
    from transformers import SamModel, SamProcessor

    if not found:
        return
    processor = SamProcessor.from_pretrained(SEGMENTER)
    model = SamModel.from_pretrained(SEGMENTER).to(device).eval()
    inputs = processor(image, input_boxes=[[f.box for f in found]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                         inputs["reshaped_input_sizes"].cpu())[0]
    best = outputs.iou_scores.cpu()[0].argmax(-1)
    for k, f in enumerate(found):
        f.mask = masks[k, int(best[k])].numpy().astype(bool)


def estimate_depth(image, setting: str = "outdoor", device: str = "cpu") -> np.ndarray:
    """Metric depth (metres along the view) at the picture's resolution."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    repo = DEPTH[setting]
    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModelForDepthEstimation.from_pretrained(repo).to(device).eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        predicted = model(**inputs).predicted_depth
    depth = torch.nn.functional.interpolate(predicted[:, None], size=image.size[::-1], mode="bicubic",
                                            align_corners=False)[0, 0]
    return depth.clamp_min(1e-3).cpu().numpy().astype(np.float64)


def overlay(image: np.ndarray, placed: list[Placed], ground: Ground, fov_deg: float) -> np.ndarray:
    """The picture with every object's mask, label and base, and a 1 m grid on the ground (uint8)."""
    from PIL import Image, ImageDraw

    out = image[..., :3].astype(np.float32).copy()
    rng = np.random.default_rng(1)
    for p in placed:
        colour = rng.uniform(60, 255, 3)
        out[p.found.mask] = out[p.found.mask] * 0.5 + colour * 0.5
    img = Image.fromarray(out.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    H, W = image.shape[:2]
    f, cx, cy = intrinsics(W, H, fov_deg)

    def project(world):
        cam = ground.origin + np.asarray(world) @ ground.axes
        if cam[2] <= 1e-6:
            return None
        return cam[0] / cam[2] * f + cx, cam[1] / cam[2] * f + cy

    extent = int(max(max((abs(c) for p in placed for c in p.position), default=1) + 2, 3))
    for k in range(-extent, extent + 1):
        for a, b in (([k, 0, -extent], [k, 0, extent]), ([-extent, 0, k], [extent, 0, k])):
            pa, pb = project(a), project(b)
            if pa and pb:
                draw.line([pa, pb], fill=(255, 255, 255), width=1)
    for p in placed:
        base = project([p.position[0], 0, p.position[1]])
        top = project([p.position[0], p.height, p.position[1]])
        if base and top:
            draw.line([base, top], fill=(255, 40, 40), width=3)
            draw.ellipse([base[0] - 5, base[1] - 5, base[0] + 5, base[1] + 5], fill=(255, 40, 40))
        x0, y0 = p.found.box[:2]
        draw.text((x0 + 3, y0 + 3), f"{p.label} {p.height:.1f} m", fill=(255, 255, 255))
    return np.asarray(img)


def analyse(image_path: Path, names: list[str], fov_deg: float = 55.0, setting: str = "outdoor",
            device: str | None = None, log=print) -> tuple[list[Placed], Ground, np.ndarray]:
    """Detect, cut out and place the named objects in a picture. Returns (placed, ground, picture)."""
    import torch
    from PIL import Image

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB")
    found = detect(image, names, device=device)
    log(f"[decompose] found {len(found)}: " + ", ".join(f"{f.label} ({f.score:.2f})" for f in found))
    if not found:
        raise ValueError(f"none of {', '.join(names)} found in {Path(image_path).name}")
    segment(image, found, device=device)
    found, dropped = distinct(found)
    for f in dropped:
        log(f"[decompose] dropped {f.label} ({f.score:.2f}): it lies inside a likelier object")
    depth = estimate_depth(image, setting, device=device)
    log(f"[decompose] depth {depth.min():.1f} to {depth.max():.1f} m")
    placed, ground = locate(found, depth, fov_deg)
    flat = [p for p in placed if p.sheet]
    if flat and len(flat) < len(placed):
        for p in flat:
            log(f"[decompose] dropped {p.label}: it lies flat along the ground, a patch of it rather than an object")
        placed, ground = locate([p.found for p in placed if p not in flat], depth, fov_deg)
    log(f"[decompose] ground plane {ground.camera_height:.1f} m below the camera; " + ", ".join(
        f"{p.label} at ({p.position[0]:.1f}, {p.position[1]:.1f}) {p.height:.1f} m" for p in placed))
    return placed, ground, np.asarray(image)


def decompose(image_path: Path, names: list[str], out: Path, name: str | None = None, style: str = "",
              fov_deg: float = 55.0, setting: str = "outdoor", known: dict[str, float] | None = None,
              each: bool = False, redraw: float = 0.0, device: str | None = None, log=print) -> dict:
    """Write a scene spec for ``image_path`` to ``out`` (with cut-outs beside it, and an overlay showing what was
    found). ``known`` gives true heights of some objects (label: metres) to correct the depth's scale."""
    from PIL import Image

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    placed, ground, image = analyse(Path(image_path), names, fov_deg, setting, device, log)
    scale = 1.0
    if known:
        ratios = [h / p.height for label, h in known.items() for p in placed if slug(p.label) == slug(label)]
        if ratios:
            scale = float(np.median(ratios))
            log(f"[decompose] scaled by {scale:.3g} to match the known heights")
    concept = out.parent / f"{out.stem}_concept{Path(image_path).suffix}"
    if Path(image_path).resolve() != concept.resolve():
        concept.write_bytes(Path(image_path).read_bytes())
    spec = to_spec(placed, image, out.parent / f"{out.stem}_parts", name or slug(out.stem), style, each, redraw,
                   scale, concept=concept.name, ground=ground, fov_deg=fov_deg)
    out.write_text(json.dumps(spec, indent=2) + "\n")
    Image.fromarray(overlay(image, placed, ground, fov_deg)).save(out.with_name(f"{out.stem}_found.png"))
    kinds, n = len(spec["objects"]), len(placed)
    log(f"[decompose] wrote {out}: {kinds} kind{'' if kinds == 1 else 's'} of object, "
        f"{n} placement{'' if n == 1 else 's'}")
    return spec
