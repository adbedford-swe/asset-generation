"""stable-diffusion.cpp's sd-server: choosing, loading and unloading its model around each picture.

sd-server holds on to its peak VRAM after a generation, which leaves too little for Idol and the
stages after it, so assetgen loads the model for each picture and unloads it afterwards.

Which model: ``ImageSpec.model`` when given, which may be
  - an option id from :func:`options` (``<recipe>:<file>``), as the web UI and Unity send it;
  - a recipe id (``flux1-schnell``), its files chosen from the server's catalog by name;
  - a model file (``checkpoints/sd15.gguf`` or just ``sd15.gguf``), its recipe chosen by name;
otherwise whatever sd-server has loaded, otherwise the model assetgen last loaded on that server,
otherwise the one model in sd-server's catalog that clearly fits a text-to-image recipe. A model
given to sd-server with ``-m`` can't be loaded back through its API, so that one is never unloaded.
"""
from __future__ import annotations

import json
import re
import time

from . import imagegen
from .imagegen import TIMEOUT, ImageSpec, ModelNotLoaded, _check, _headers, sdapi

# Which catalog categories can fill which recipe slot (sd-server: model_catalog.cpp, k_model_slots).
SLOT_CATEGORIES = {
    "model": ("checkpoints",), "diffusion_model": ("diffusion_models",),
    "high_noise_diffusion_model": ("diffusion_models",), "uncond_diffusion_model": ("diffusion_models",),
    "clip_l": ("text_encoders",), "clip_g": ("text_encoders",), "t5xxl": ("text_encoders",),
    "llm": ("text_encoders",), "llm_vision": ("text_encoders", "clip_vision"), "tokenizer": ("tokenizers",),
    "vae": ("vae",), "tae": ("vae", "vae_approx"), "taesd": ("vae_approx", "vae"), "clip_vision": ("clip_vision",),
    "audio_vae": ("vae",), "audio_encoder": ("text_encoders",), "embeddings_connectors": ("text_encoders",),
    "control_net": ("controlnet",), "ip_adapter": ("ipadapter",), "motion_module": ("motion_modules",),
    "photo_maker": ("photomaker",), "pulid_weights": ("pulid",),
}
PRIMARY_SLOTS = ("model", "diffusion_model")
# Recipes for something other than plain text to image.
NOT_TEXT_TO_IMAGE = re.compile(r"edit|inpaint|kontext|fill|canny|depth|control|ip-adapter|photomaker|pulid|pix2pix|"
                               r"animatediff|video|i2v|t2v|wan|ltx|hunyuan|upscale")
# File names that say which family a checkpoint belongs to, where the recipe id alone doesn't.
FAMILY_HINTS = {
    "sd1-checkpoint": ("sd1", "sd15", "v15", "sdv1", "dreamshaper8", "realisticvision"),
    "sd2-768-checkpoint": ("sd21", "v21", "sd2768"),
    "sdxl-checkpoint": ("sdxl", "ssd1b", "segmindvega", "juggernautxl", "pony", "illustrious", "noobai"),
}


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _request(spec: ImageSpec, method: str, path: str, timeout: float = 30, **kwargs):
    import requests

    return requests.request(method, f"{spec.base_url}{path}", headers=_headers(spec.backend), timeout=timeout,
                            **kwargs)


def status(spec: ImageSpec) -> dict | None:
    """sd-server's model status, or None from a server without the model API (an older sd-server)."""
    r = _request(spec, "GET", "/sdcpp/v1/models/status", timeout=10)
    return r.json() if r.ok else None


def catalog(spec: ImageSpec) -> tuple[list[dict], list[dict]]:
    """sd-server's recipes and model files."""
    recipes = _check(_request(spec, "GET", "/sdcpp/v1/recipes", timeout=30)).json().get("recipes", [])
    files = _check(_request(spec, "GET", "/sdcpp/v1/model-files", timeout=30)).json().get("files", [])
    return recipes, files


def _name_score(recipe: dict, file: dict) -> int:
    """How clearly a file's name says it belongs to a recipe: the length of the longest key of the recipe
    (its id, without -checkpoint, or a family hint) found in the name. 0 when none is."""
    name = _normalise(file["id"])
    keys = {_normalise(recipe["id"].replace("-checkpoint", ""))} | set(FAMILY_HINTS.get(recipe["id"], ()))
    keys |= {_normalise(recipe.get("name", ""))} - {""}
    return max((len(k) for k in keys if len(k) >= 3 and k in name), default=0)


def _pick(slot: str, recipe: dict, files: list[dict], taken: set[str]) -> dict | None:
    """The file to fill a secondary slot: in the slot's categories, preferring names that mention the slot
    (clip_l, t5xxl, ae for a FLUX VAE) and then the recipe."""
    categories = SLOT_CATEGORIES.get(slot, ())
    pool = [f for f in files if f["category"] in categories and f["id"] not in taken]
    if not pool:
        return None
    slot_keys = {_normalise(slot), *{"vae": ("vae", "ae"), "t5xxl": ("t5",), "llm": ("qwen", "llama", "gemma")}
                 .get(slot, ())}

    def score(f):
        name = _normalise(f["id"])
        return (max((len(k) for k in slot_keys if k in name), default=0), _name_score(recipe, f))

    return max(pool, key=score)


def _fill(recipe: dict, primary: dict, files: list[dict]) -> dict | None:
    """A complete model selection for ``recipe`` built around ``primary``, or None when a required slot
    has no file."""
    slots = recipe.get("slots", [])
    primary_slot = next((s["id"] for s in slots if s["id"] in PRIMARY_SLOTS and s.get("required")), None)
    if primary_slot is None or primary["category"] not in SLOT_CATEGORIES[primary_slot]:
        return None
    models = {primary_slot: primary["id"]}
    for slot in slots:
        if slot["id"] == primary_slot or not slot.get("required"):
            continue
        if slot["id"] in (recipe.get("models") or {}):
            continue  # the recipe brings its own default for it
        chosen = _pick(slot["id"], recipe, files, set(models.values()))
        if chosen is None:
            return None
        models[slot["id"]] = chosen["id"]
    for group in recipe.get("required_any", []):
        if any(models.get(s) or (recipe.get("models") or {}).get(s) for s in group):
            continue
        chosen = next((c for c in (_pick(s, recipe, files, set(models.values())) for s in group) if c), None)
        if chosen is None:
            return None
        models[next(s for s in group if chosen["category"] in SLOT_CATEGORIES.get(s, ()))] = chosen["id"]
    return models


def options(spec: ImageSpec, every_recipe: bool = False) -> list[dict]:
    """The text-to-image models the server's files can load, best name match first:
    ``{"id": "<recipe>:<file>", "label", "recipe", "models", "score"}``. For each file only the recipes its
    name matches best, unless none matches or ``every_recipe``."""
    recipes, files = catalog(spec)
    found = []
    for recipe in recipes:
        if NOT_TEXT_TO_IMAGE.search(recipe["id"]) or "video_frames" in (recipe.get("defaults") or {}):
            continue
        for primary in files:
            models = _fill(recipe, primary, files)
            if models is None:
                continue
            found.append({"id": f"{recipe['id']}:{primary['id']}",
                          "label": f"{primary.get('name', primary['id'])} · {recipe.get('name', recipe['id'])}",
                          "recipe": recipe["id"], "models": models, "score": _name_score(recipe, primary)})
    if not every_recipe:
        best: dict[str, int] = {}
        for o in found:
            file = o["id"].split(":", 1)[1]
            best[file] = max(best.get(file, 0), o["score"])
        found = [o for o in found if o["score"] == best[o["id"].split(":", 1)[1]]]
    found.sort(key=lambda o: (-o["score"], o["label"]))
    return found


def automatic(found: list[dict]) -> dict | None:
    """The option to load without being told: the only one there is, or the only best name match for each
    model file when there is one file. None when it isn't clear."""
    if len(found) == 1:
        return found[0]
    by_file: dict[str, list[dict]] = {}
    for o in found:
        by_file.setdefault(o["id"].split(":", 1)[1], []).append(o)
    best = []
    for group in by_file.values():
        top = [o for o in group if o["score"] == group[0]["score"] and o["score"] > 0]
        if len(top) == 1:
            best.append(top[0])
    return best[0] if len(best) == 1 and len(by_file) == 1 else None


def selection_for(spec: ImageSpec, found: list[dict] | None = None) -> dict:
    """The model selection ``spec.model`` names (see the module docstring)."""
    wanted = spec.model or ""
    found = found if found is not None else options(spec, every_recipe=True)
    exact = next((o for o in found if o["id"] == wanted), None)
    if exact:
        return {"recipe": exact["recipe"], "models": exact["models"]}
    by_recipe = [o for o in found if o["recipe"] == wanted]
    if by_recipe:
        return {"recipe": wanted, "models": by_recipe[0]["models"]}
    by_file = sorted((o for o in found if _normalise(o["id"].split(":", 1)[1]).endswith(_normalise(wanted))),
                     key=lambda o: -o["score"])
    if by_file:
        return {"recipe": by_file[0]["recipe"], "models": by_file[0]["models"]}
    if ":" in wanted:  # a recipe:file the catalog did not offer; let sd-server judge it
        recipe, file = wanted.split(":", 1)
        return {"recipe": recipe, "models": {}} if not file else {"recipe": recipe, "models": {"model": file}}
    return {"recipe": wanted, "models": {}}


def _remembered(spec: ImageSpec) -> dict | None:
    try:
        return json.loads((imagegen.STATE_DIR / "sdcpp_models.json").read_text()).get(spec.base_url)
    except (OSError, ValueError):
        return None


def _remember(spec: ImageSpec, selection: dict) -> None:
    path = imagegen.STATE_DIR / "sdcpp_models.json"
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        state = {}
    if state.get(spec.base_url) != selection:
        state[spec.base_url] = selection
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2))


def _no_model_message(spec: ImageSpec, found: list[dict]) -> str:
    if not found:
        return (f"sd-server at {spec.base_url} has no model loaded and none in its model folders fits a "
                "text-to-image recipe: put a model in its models/checkpoints or models/diffusion_models folder")
    listed = "\n".join(f"  {o['id']}   ({o['label']})" for o in found[:12])
    return (f"sd-server at {spec.base_url} has no model loaded and assetgen can't tell which to load. Choose one in "
            f"the web UI or Unity, or pass --image-model with one of:\n{listed}")


def load(spec: ImageSpec, log=print) -> dict | None:
    """Make sure sd-server has the model to generate with. Returns the selection that can be loaded back
    after an unload, or None when there is none (a model from sd-server's command line, or a server
    without the model API)."""
    target = None
    for _ in range(120):
        current = status(spec)
        if current is None:
            return None
        state, recipe = current.get("state"), current.get("recipe")
        if state == "loading":
            time.sleep(1.0)
            continue
        loaded = {"recipe": recipe, "models": current.get("models") or {}} if isinstance(recipe, str) else None
        if state == "ready" and loaded is None and current.get("origin") == "command_line" and not spec.model:
            return None
        if target is None:
            if spec.model:
                target = selection_for(spec)
            elif state == "ready" and loaded:
                target = loaded
            else:
                target = _remembered(spec)
                if target is None:
                    found = options(spec)
                    chosen = automatic(found)
                    if chosen is None:
                        raise RuntimeError(_no_model_message(spec, found))
                    target = {"recipe": chosen["recipe"], "models": chosen["models"]}
        if state == "ready" and loaded and loaded["recipe"] == target["recipe"] and \
                all(loaded["models"].get(k) == v for k, v in target["models"].items()):
            _remember(spec, loaded)
            return loaded
        log(f"[image] loading {target['recipe']} ({', '.join(target['models'].values()) or 'recipe defaults'}) "
            f"on sd-server")
        r = _request(spec, "POST", "/sdcpp/v1/models/load", json=target, timeout=TIMEOUT)
        if r.status_code in (409, 503):  # another job is switching models or generating
            time.sleep(2.0)
            continue
        if r.status_code == 422 and not spec.model and target == _remembered(spec):
            log(f"[image] the model assetgen remembered no longer loads ({r.text[:200]}); choosing again")
            _forget(spec)
            target = None
            continue
        _check(r)
        _remember(spec, target)
        return target
    raise TimeoutError(f"sd-server at {spec.base_url} stayed busy for several minutes")


def _forget(spec: ImageSpec) -> None:
    path = imagegen.STATE_DIR / "sdcpp_models.json"
    try:
        state = json.loads(path.read_text())
        state.pop(spec.base_url, None)
        path.write_text(json.dumps(state, indent=2))
    except (OSError, ValueError):
        pass


def unload(spec: ImageSpec, log=print) -> bool:
    """Unload sd-server's model, freeing its VRAM. False when it is busy with another generation."""
    r = _request(spec, "POST", "/sdcpp/v1/models/unload", json={}, timeout=120)
    if r.status_code in (409, 503):
        log("[image] sd-server is busy with another generation; its model stays loaded")
        return False
    _check(r)
    log("[image] unloaded sd-server's model")
    return True


def generate(prompt: str, spec: ImageSpec, seed: int, count: int) -> list[bytes]:
    """Load the model if needed, generate, unload."""
    return with_model(spec, lambda: sdapi(prompt, spec, seed, count))


def with_model(spec: ImageSpec, call):
    """``call()`` with sd-server's model loaded: load it if needed, and unload it after."""
    for _ in range(3):
        selection = load(spec)
        try:
            return call()
        except ModelNotLoaded:
            continue  # another job unloaded it between the load and the generation
        finally:
            if not spec.keep_loaded:
                if selection is not None:
                    unload(spec)
                elif (status(spec) or {}).get("origin") == "command_line":
                    print("[image] sd-server's model came from its command line (-m), so assetgen cannot load it "
                          "back and leaves it loaded; start sd-server with --model-dir instead to free the GPU "
                          "between pictures")
    raise ModelNotLoaded(f"sd-server at {spec.base_url} kept losing its model while generating")


def choices(spec: ImageSpec) -> dict:
    """What a model picker shows: the model sd-server has loaded, every model it could load, and the one
    assetgen would load by itself. Never raises; ``error`` says why there are no options."""
    out = {"loaded": None, "options": [], "automatic": None, "error": None}
    try:
        current = status(spec)
        if current is None:
            out["error"] = "this sd-server has no model API (update stable-diffusion.cpp)"
            return out
        if current.get("state") == "ready":
            out["loaded"] = {"recipe": current.get("recipe"), "models": current.get("models") or {},
                             "origin": current.get("origin")}
        found = options(spec)
        out["options"] = [{k: o[k] for k in ("id", "label", "recipe", "models")} for o in found]
        remembered = _remembered(spec)
        pick = automatic(found)
        if remembered:
            pick = next((o for o in found if o["recipe"] == remembered["recipe"]
                         and o["models"] == remembered["models"]), pick)
        out["automatic"] = pick["id"] if pick else None
        if not found:
            out["error"] = _no_model_message(spec, found)
    except Exception as e:  # shown to the user, not raised
        out["error"] = f"{type(e).__name__}: {e}"
    return out
