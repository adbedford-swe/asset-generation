# Changelog

## 0.4.0

- Runs the assetgen server itself: starts it when you generate and none answers, shows its state in the window
  (Start, Stop, Restart, Log), stops it with the Editor. Local server settings in Project Settings > Assetgen, and
  `unity command assetgen.server start | stop | status | log`.
- Generation settings (Idol and motion steps, prompt strength, seed, loop length, root motion, mesh and texture
  budgets, rig candidates) in the window's Advanced section and Project Settings, sent with every job;
  `assetgen generate` takes `--idol-steps`, `--motion-steps` and `--seed`.

## 0.3.0

- First release: `.assetgen` importer, Animator Controller and prefab builder with a speed-driven locomotion blend
  tree, `AssetgenCharacter`, the Assetgen window and project settings, and `unity command assetgen` commands for the
  Unity CLI.
