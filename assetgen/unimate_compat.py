"""Run a UniMate entry point with the fixes this environment needs, leaving UniMate unchanged.

* ``--blender``: UniMate targets bpy 4.0. Since Blender 4.4 an action is bound to
  an object through an action slot, and UniMate assigns its rebuilt action before
  creating the F-curves (which is what creates the slot), so the armature is left
  on no slot: nothing moves and the glTF exporter writes no animation. The slot is
  assigned once the action is built.
* Otherwise (the sampler): the released checkpoint's config samples on ``cuda``;
  without a GPU, sample on the CPU instead. Sampling goes through
  ``assetgen.motion_sampler`` (fixed-step solve, seamless loops).

    python -m assetgen.unimate_compat [--blender] <script.py | -m module> [args...]
"""
from __future__ import annotations

import os
import runpy
import sys
import traceback


def _patch_device() -> None:
    import torch
    from unimate.configs.schema import MainConfig

    if torch.cuda.is_available():
        return
    from_json = MainConfig.from_json.__func__

    @classmethod
    def cpu_from_json(cls, config_path):
        config = from_json(cls, config_path)
        config.sampling.device = "cpu"
        return config

    MainConfig.from_json = cpu_from_json


def _patch_blender() -> None:
    import data_process.utils.blender_rig as blender_rig

    rebuild = blender_rig.rebuild_action_from_data

    def rebuild_action_from_data(armature, anim_data_dict):
        rebuild(armature, anim_data_dict)
        ad = armature.animation_data
        if ad is not None and ad.action is not None and hasattr(ad, "action_slot") and ad.action_slot is None:
            if len(ad.action.slots):
                ad.action_slot = ad.action.slots[0]

    blender_rig.rebuild_action_from_data = rebuild_action_from_data


def _run(args: list[str]) -> None:
    if args[0] == "-m":
        sys.argv = [args[1], *args[2:]]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = args
        runpy.run_path(args[0], run_name="__main__")


def main() -> None:
    args = sys.argv[1:]
    if args[0] != "--blender":
        _patch_device()
        from .motion_sampler import install

        install()
        _run(args)
        return
    # bpy as a module tears itself down badly: it reports leaked blocks, then crashes (Windows) or
    # hangs (Linux) at interpreter exit, after the script's files are written. Leave without teardown.
    _patch_blender()
    code = 0
    try:
        _run(args[1:])
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if os.name == "nt":
            # os._exit still unloads DLLs on Windows, which runs Blender's static destructors.
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel32.TerminateProcess(kernel32.GetCurrentProcess(), code)
        os._exit(code)


if __name__ == "__main__":
    main()
