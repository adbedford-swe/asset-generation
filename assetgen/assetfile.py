"""The ``.assetgen`` file beside every finished asset, whatever its kind.

Its ``format`` says what it describes: ``assetgen.character`` (:mod:`assetgen.character`)
or ``assetgen.prop`` for props and dioramas (:mod:`assetgen.prop`). Engines and the
server read it to know how to set the asset up.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import character, prop

EXTENSION = ".assetgen"
READERS = {character.FORMAT: character.load, prop.FORMAT: prop.load}


def path_for(glb: Path) -> Path:
    return Path(glb).with_suffix(EXTENSION)


def load(path: Path) -> dict:
    """Any assetgen file, checked against its own format's reader."""
    data = json.loads(Path(path).read_text())
    reader = READERS.get(data.get("format"))
    if reader is None:
        raise ValueError(f"{path} is not an assetgen file (format {data.get('format')!r})")
    return reader(path)


def ensure(glb: Path) -> Path:
    """The file beside ``glb``. Character files are written now for runs finalized before they existed;
    props have always had theirs."""
    path = path_for(glb)
    if not path.exists():
        character.write(glb)
    return path
