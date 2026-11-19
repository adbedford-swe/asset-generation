"""Scenes: environments made of generated props, laid out on generated ground.

:mod:`.spec` says what a scene holds, :mod:`.build` makes it (objects through the prop
pipeline, then :mod:`.terrain`, :mod:`.layout`, :mod:`.harmonize` and :mod:`.assemble`).
"""
from .spec import SceneSpec, SpecError, load, parse

__all__ = ["SceneSpec", "SpecError", "load", "parse"]
