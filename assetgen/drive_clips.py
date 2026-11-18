"""Drive the canonical asset with every sampled motion in one Blender session.

UniMate's ``animate_motion.py`` handles one motion per process and always writes GLB and
FBX; starting Blender per clip and exporting FBX nobody asked for dominated the motion
stage after sampling. This runs its ``animate_character`` for each job in turn (it resets
the scene first) and exports only the requested formats.

    python -m assetgen.unimate_compat --blender -m assetgen.drive_clips --jobs jobs.json [--formats glb,fbx]
"""
from __future__ import annotations

import argparse
import functools
import json
import time


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", required=True, help="JSON list of {anim, char, cond, dtype, out_dir}")
    ap.add_argument("--formats", default="glb")
    a = ap.parse_args(argv)

    import data_process.mesh_animation.animate_motion as am

    formats = tuple(f for f in a.formats.split(",") if f)
    am.drive_and_export = functools.partial(am.drive_and_export, formats=formats)
    jobs = json.loads(open(a.jobs, encoding="utf-8").read())
    for i, job in enumerate(jobs, 1):
        t0 = time.time()
        am.animate_character(char_path=job["char"], anim_path=job["anim"], cond_path=job["cond"],
                             output_dir=job["out_dir"], dataset_type=job["dtype"], anim_mode="fk",
                             extra_bones_strategy="merge")
        print(f"[drive] {i}/{len(jobs)} {job['anim']} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
