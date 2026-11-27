# Assetgen for Unity

Describe a character, or hand over a picture, and get back a prefab that walks, runs and attacks. Or describe a
prop (a barrel, a tree) or a diorama (a whole set piece) and get back a prefab with its collider, or a whole scene
and get back an environment to walk around in.

This package is the Unity end of [assetgen](https://github.com/adbedford-swe/asset-generation). It imports
assetgen characters, props, dioramas and scenes ready to use, generates new ones from an assetgen server without leaving
the Editor, and adds `assetgen` commands to the Unity CLI so an agent can do the same.

## Install

Unity 6 (6000.0 or later). In **Window > Package Manager > + > Install package from git URL**:

```
https://github.com/adbedford-swe/asset-generation.git?path=/unity/com.assetgen.unity
```

or let assetgen add it while exporting: `assetgen export <run> --engine unity --project <your project>`.
glTF import comes from Unity's own glTFast (`com.unity.cloud.gltfast`), installed as a dependency.

## What an imported character is

Drop `fox.glb` and `fox.assetgen` (an assetgen export) anywhere under `Assets/`, and the package builds:

| Asset | What it holds |
|---|---|
| `fox.assetgen` | The character definition: every clip's role (idle, walk, run, action), loop, length and speed. |
| `fox.controller` | An Animator Controller. **Locomotion** is a 1D blend tree over idle, walk and run, on a `Speed` parameter in metres per second, at the speeds the clips were generated at. Every clip also has its own state, and one-shots fade back to Locomotion when they end. |
| `fox.prefab` | A variant of the model, with that controller and an `AssetgenCharacter` component. |

Re-importing updates the controller and prefab in place, so scenes and references keep working.

```csharp
var fox = GetComponent<AssetgenCharacter>();
fox.Speed = fox.WalkSpeed;      // the blend tree plays the walk at its own pace: no foot sliding
fox.Play("jumps_in_place");     // one-shots return to Locomotion by themselves
fox.Locomote(fox.RunSpeed);     // back to moving, now at a run
```

Clips play in place: the character file stores how fast each one travels, so your controller, NavMeshAgent or
`AssetgenCharacter.MoveForward` moves the character, and the Animator never fights it. Above the fastest clip's
speed, `AssetgenCharacter` speeds the animation up to keep the feet planted.

Clips made from video (motion prompts tagged `[video]`) carry their timing, which the builder turns into
animation events, and their sound, which plays from an AudioSource it adds to the prefab:

```csharp
fox.Footstep += bone => footstepAudio.Play();                  // part of the body came to rest on the ground
fox.Impact += clip => hitBox.Check();                          // a one-shot's action lands
var phase = fox.PhaseOf("pounces_on_something", 0.6f);         // "windup", "active" or "recovery"
```

## What an imported prop or diorama is

A prop's `.assetgen` file says it is static (format `assetgen.prop`). Drop `barrel.glb` and `barrel.assetgen` under
`Assets/` and the package builds:

| Asset | What it holds |
|---|---|
| `barrel.assetgen` | The prop definition: prop or diorama, height and size, collider, and footprint (its outline seen from above, its radius, and the radius of its base). |
| `barrel.prefab` | A variant of the model with its collider and an `AssetgenProp` component. Props get a convex `MeshCollider` on each mesh; dioramas a non-convex one (you walk on those) and are marked static; a file can also ask for one `BoxCollider` or none. |

The model's origin is the centre of its base, on the ground, so it stands where you place it. Re-importing updates
the prefab in place; colliders added to it by hand are replaced. Layout code can space props by their footprint:

```csharp
var barrel = GetComponent<AssetgenProp>();
var clearance = barrel.FootprintRadius;          // world units, at the prop's scale
var outline = barrel.Definition.outline;         // (x, z) points around the origin, seen from above
```

## What an imported scene is

A scene is a ground and objects laid out on it, all in one model, with each object also on its own as a prop.
Drop the export (`camp.glb`, `camp.assetgen` and its `assets/` folder) under `Assets/` and the package builds:

| Asset | What it holds |
|---|---|
| `camp.assetgen` | The scene definition: the ground, each kind of object (its prop file, height, collider and footprint) and every placement (node, position, rotation, scale). |
| `camp.prefab` | A variant of the scene model, all static: a `MeshCollider` on the ground, and on each placement the collider its object asks for (convex or non-convex mesh colliders, or a box), with an `AssetgenScene` component. |
| `assets/pine.prefab`, ... | Each object as a prop (see above), to place more of by hand. |

Each object's mesh is stored once in the model and every placement points at it. `AssetgenScene` says which object
a placement, or a collider a ray hit, is:

```csharp
var camp = GetComponent<AssetgenScene>();
foreach (var pine in camp.InstancesOf("pine")) { /* ... */ }
if (Physics.Raycast(ray, out var hit))
    Debug.Log(camp.ObjectOf(hit.transform)?.id ?? "the ground");
```

## Generating from the Editor

**Window > Assetgen.** Pick Character, Prop or Diorama, describe it or pick a picture, choose a character's motions
(or a preset) and press **Generate**. Progress shows per stage, and the asset is imported into
`Assets/Characters/<name>/` (props and dioramas: `Assets/Props/<name>/`) when it finishes, even if you close the
window. **Scene** takes a description, which the server's language model plans into a scene (start the server with
`--llm-base-url`), or a scene spec file, and optionally an earlier scene whose objects to reuse; scenes are imported
into `Assets/Environments/<name>/`. **Advanced** holds the image model (the models the server's image
generator can load, with Automatic first), Idol's steps, the motion sampler's steps, prompt strength, seed, loop
length, root motion, mesh and texture budgets and rig candidates. They are saved with the project, so the window and
the CLI generate the same way.

**The server runs itself.** When nothing answers at the server URL, Generate starts `server.py` in the background (no
console window, logging to `Library/Assetgen/server.log`) and waits until it's ready. The toolbar shows its state, with
Start, Stop, Restart and Log, and the server Unity started stops when the Editor quits. A server you started yourself
shows as *external* and is left alone. Unity finds the assetgen checkout when the package was installed from it (as
`assetgen export --project` does). Otherwise, in **Project Settings > Assetgen > Local server**, set the folder, the
Python (by default the checkout's `.venv`), any environment such as `ASSETGEN_THIRD_PARTY`, and extra server arguments
such as `--image-backend sdcpp --image-url http://127.0.0.1:1234`. These are saved per machine, not in the project.

## From the Unity CLI and agents

With the [Unity CLI](https://docs.unity.com/en-us/unity-cli) and its Pipeline package (`unity pipeline install`)
in the project, the package adds an `assetgen` command group to the running Editor:

```bash
unity command assetgen status
unity command assetgen generate --prompt "a goblin shaman with a lantern" \
    --motions '["An object stands idle.","An object walks forward.","An object casts a spell. [once]"]' \
    --wait true --timeout 1800
unity command assetgen spawn --character a_goblin_shaman_with_a_lantern --position 0,0,2
unity command assetgen generate --kind prop --prompt "a wooden barrel with iron bands" --height 1 --wait true --timeout 1800
unity command assetgen scene --spec camp.json --wait true --timeout 3600
unity command editor.playmode start
unity command assetgen play --target a_goblin_shaman_with_a_lantern --speed 0.5
```

| Command | Does |
|---|---|
| `status` | The server, its devices and image backend, and the characters, props and scenes in the project. |
| `generate` | Starts a character from `--prompt` and/or `--image`, starting the server if none answers; `--kind prop` or `--kind diorama` makes a static asset instead. `--idol-steps`, `--motion-steps` and `--seed` override the project's settings. Returns the job id; with `--wait`, waits and returns the prefab. |
| `scene` | Starts a scene from `--prompt` (planned by the server's language model) or `--spec` (a scene spec file or JSON); `--from` reuses an earlier scene's objects. Returns the job id; with `--wait`, waits and returns the prefab. |
| `wait --id` | Waits for a job, imports it, returns the prefab and its clips. |
| `job --id` / `cancel --id` | A job's progress; stop it. |
| `import --source` | A job id, an export zip, a folder or a `.glb`/`.assetgen` pair into the project. |
| `list` | Every character in the project, with prefab and clips; every prop and diorama, with its kind, size and collider; every scene, with its objects. |
| `image_models` | The text-to-image models the server can load, for `generate --image-model`. |
| `build --character` | Rebuild a character's controller and prefab, or a prop's or scene's prefab. |
| `spawn --character` | Put a character, prop, diorama or generated scene in the open scene (`--position`; for characters `--speed`, `--move-forward`). |
| `play --target` | In Play mode: play a clip, or set the locomotion speed. |
| `assetgen.server start` / `stop` / `status` / `log` | Run the local server from the project's settings, stop the one Unity started, check it, read its log. |

`unity mcp` serves the same commands as MCP tools (`assetgen_generate`, `assetgen_spawn`, ...), so any MCP client
can go from a sentence to a character standing in the scene. The commands live in an assembly that compiles only
when `com.unity.pipeline` is present; without it, everything else works as usual.

## Licence of generated motion

assetgen animates with UniMate's released checkpoints, which are CC BY-NC 4.0: characters animated with them are for
non-commercial use. Clips made from video are tracked with CoTracker3, also CC BY-NC 4.0. The character file's
`license` field says so, and so does its inspector.
