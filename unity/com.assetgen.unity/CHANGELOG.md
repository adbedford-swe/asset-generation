# Changelog

## 0.7.0

- Clips made from video carry their timing. The builder adds animation events for their footsteps
  and a one-shot's impact, and `AssetgenCharacter` raises them as `Footstep`, `Impact` and `ClipEvent`;
  `PhaseOf(clip, time)` says whether a one-shot is winding up, active or recovering.
- Their sounds (`sounds/` beside the `.assetgen` file) go on the prefab's `AssetgenCharacter` and play from
  an AudioSource the builder adds, as each clip starts.
- `CharacterDefinition.Clip` has the clip's `source`, `events`, `phases` and `sound`.

## 0.6.0

- Scenes: `.assetgen` files of format `assetgen.scene` import as a `SceneDefinition` (ground, objects, placements),
  and build `name.prefab`, a variant of the scene model with a MeshCollider on the ground, each placement's collider
  as its object asks, everything static, and an `AssetgenScene` component mapping placements to objects. Each object
  in the scene's `assets/` folder is built as a prop too. Scenes are imported into a new environments folder
  (`Assets/Environments` by default).
- Export zips keep their folders (a scene's `assets/`), still refusing paths out of the asset's folder.
- The window has a Scene mode (a description or a spec file, and an earlier scene to reuse objects from), and lists
  scenes in the project. `unity command assetgen scene --prompt | --spec`; `import`, `build`, `spawn`, `list`, `wait`
  and `status` handle scenes too.

## 0.5.0

- Props and dioramas: `.assetgen` files of format `assetgen.prop` import as a `PropDefinition` (kind, size, collider,
  footprint), and build `name.prefab`, a variant of the model with its collider (convex or non-convex mesh colliders,
  or a box) and an `AssetgenProp` component. Dioramas are marked static. They are imported into a new props folder
  (Project Settings > Assetgen, `Assets/Props` by default).
- The window generates a character, a prop or a diorama (motions only for characters), and lists props in the project.
- `unity command assetgen generate --kind prop|diorama`; `import`, `build`, `spawn`, `list` and `status` handle props
  too.

## 0.4.1

- Image model picker in the window's Advanced section and Project Settings (Automatic, or any model the server's
  image generator can load), `unity command assetgen image_models`, and `generate --image-model`.

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
