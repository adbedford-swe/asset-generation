"""From a description to a scene spec, with a language model.

"a forest camp at dusk with a tent, a campfire and logs to sit on" becomes a spec:
the objects that make it up (each described on its own, at its real height), where
the important ones go, what is scattered around them, and a style shared by every
prompt. Any OpenAI-compatible chat endpoint does (llama.cpp's server, vLLM, Ollama,
OpenAI, ...). The answer is checked like any spec; if it does not hold up, the model
is told what is wrong and asked again.

    assetgen scene plan --text "a forest camp at dusk" --llm-base-url http://127.0.0.1:8080/v1 --out camp.json
"""
from __future__ import annotations

import json
import os
import re

from .spec import SpecError, parse

EXAMPLE = {
    "name": "forest_camp",
    "style": "stylised low-poly, hand-painted textures, warm evening light",
    "seed": 7,
    "ground": {"size": [24, 24], "prompt": "mossy forest floor with pine needles", "relief": 0.6},
    "objects": [
        {"id": "campfire", "prompt": "a ring of stones around burning logs", "height": 0.5,
         "place": [{"position": [0, 0]}]},
        {"id": "tent", "prompt": "a canvas tent with wooden poles", "height": 2.2,
         "place": [{"position": [3.5, -2], "yaw": {"face": "campfire"}}]},
        {"id": "log_bench", "prompt": "a fallen log used as a bench", "height": 0.5,
         "place": [{"position": [-2, 0.8], "yaw": {"face": "campfire"}},
                   {"position": [0.5, 2.2], "yaw": {"face": "campfire"}}]},
        {"id": "pine", "prompt": "a tall pine tree", "height": 9,
         "scatter": {"count": 16, "region": "edge", "scale": [0.8, 1.25]}},
        {"id": "boulder", "prompt": "a mossy granite boulder", "height": 1.0,
         "scatter": {"count": 8, "scale": [0.5, 1.4], "tilt": True}},
    ],
}

INSTRUCTIONS = """You plan 3D game environments for a generator that makes every object from a short description \
and lays them out on a ground. Answer with one JSON object, nothing else, in this format:

{example}

Rules:
- "objects" lists the distinct kinds of object, 2 to {max_objects} of them. Each is generated once and placed as \
often as needed, so list a tree once and scatter it.
- Each "prompt" describes one object on its own, as it would be photographed alone: what it is, its shape and \
material. No scene, no surroundings, no lighting, no other objects, and do not repeat the style.
- "height" is the object's real height in metres.
- "place" puts the objects that matter at chosen spots: "position" is [x, z] in metres, with (0, 0) the middle of \
the ground and x and z within half the ground's size. "yaw" turns an object: degrees, "random", or \
{{"face": "<id>"}} to turn its front towards another object.
- "scatter" spreads repeated things (trees, rocks, bushes, debris): "count", "region" (all, edge, centre, or \
[[x0, z0], [x1, z1]]), "scale" [smallest, largest], and "tilt": true for things that lean with the slope.
- Give each object exactly one of "place" or "scatter". Keep the counts sensible for the ground's size.
- "style" is the look shared by every object: art style, texture style, lighting. One line.
- "ground": "size" [width, depth] in metres, "prompt" is the ground surface's material (seen from above), \
"relief" is the height of its hills in metres (0 for floors and streets).
- "id" values are lowercase words joined by _, unique, starting with a letter. "name" likewise."""


def extract_json(text: str) -> dict:
    """The JSON object in a model's answer, which may come wrapped in a code fence or prose."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise SpecError("the answer holds no JSON object")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise SpecError(f"the answer is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise SpecError("the answer is not a JSON object")
    return data


def _chat(base_url: str, model: str, messages: list[dict], json_mode: bool, timeout: float) -> str:
    import requests

    base = base_url.rstrip("/")
    base = base if base.endswith("/v1") else f"{base}/v1"
    key = os.environ.get("ASSETGEN_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    body = {"model": model, "messages": messages, "temperature": 0.4}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(f"{base}/chat/completions", json=body, timeout=timeout,
                      headers={"Authorization": f"Bearer {key}"} if key else {})
    if r.status_code >= 400:
        raise RuntimeError(f"{base}/chat/completions -> {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"] or ""


def plan(description: str, base_url: str, model: str = "local", size: float | None = None, max_objects: int = 8,
         attempts: int = 3, timeout: float = 300, log=print) -> dict:
    """A checked scene spec (as a dict) for ``description``."""
    system = INSTRUCTIONS.format(example=json.dumps(EXAMPLE, indent=1), max_objects=max_objects)
    ask = description.strip()
    if size:
        ask += f"\n\nThe ground is {size:g} by {size:g} metres."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": ask}]
    json_mode = True
    for attempt in range(1, attempts + 1):
        try:
            answer = _chat(base_url, model, messages, json_mode, timeout)
        except RuntimeError as e:
            if not (json_mode and "-> 400" in str(e)):
                raise
            json_mode = False  # some servers refuse response_format; the instructions still ask for JSON
            answer = _chat(base_url, model, messages, json_mode, timeout)
        try:
            data = extract_json(answer)
            parse(data)
            log(f"[plan] {len(data['objects'])} kinds of object: {', '.join(o['id'] for o in data['objects'])}")
            return data
        except SpecError as e:
            log(f"[plan] attempt {attempt}: {e}")
            messages += [{"role": "assistant", "content": answer},
                         {"role": "user", "content": f"That does not work: {e}. Answer again with the whole, "
                                                     f"corrected JSON object only."}]
    raise SpecError(f"the model did not produce a usable scene spec in {attempts} attempts")
