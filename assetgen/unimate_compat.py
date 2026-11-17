"""Run a UniMate entry point under bpy 4.5.

UniMate targets bpy 4.0. Since Blender 4.4 an action is bound to an object
through an action slot, and UniMate assigns its rebuilt action before creating
the F-curves (which is what creates the slot), so the armature is left on no
slot: nothing moves and the glTF exporter writes no animation. Assign the slot
once the action is built, then run the requested script or module unchanged.

    python -m assetgen.unimate_compat <script.py | -m module> [args...]
"""
from __future__ import annotations

import runpy
import sys


def _patch() -> None:
    import data_process.utils.blender_rig as blender_rig

    rebuild = blender_rig.rebuild_action_from_data

    def rebuild_action_from_data(armature, anim_data_dict):
        rebuild(armature, anim_data_dict)
        ad = armature.animation_data
        if ad is not None and ad.action is not None and hasattr(ad, "action_slot") and ad.action_slot is None:
            if len(ad.action.slots):
                ad.action_slot = ad.action.slots[0]

    blender_rig.rebuild_action_from_data = rebuild_action_from_data


def main() -> None:
    args = sys.argv[1:]
    _patch()
    if args[0] == "-m":
        sys.argv = [args[1], *args[2:]]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = args
        runpy.run_path(args[0], run_name="__main__")


if __name__ == "__main__":
    main()
