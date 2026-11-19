using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.Pipeline.Commands;

namespace Assetgen.Editor
{
    /// <summary>
    /// <c>unity command assetgen ...</c>: generate characters, props and dioramas, import them and put
    /// them in scenes from the Unity CLI, or from an agent through <c>unity mcp</c> (each command is the
    /// tool <c>assetgen_&lt;name&gt;</c>). Compiled only when com.unity.pipeline is in the project.
    /// </summary>
    static class AssetgenCommands
    {
        const string Group = "assetgen";
        const string ServerGroup = "assetgen.server";

        [CliGroupCommand(Group, "The assetgen server this project talks to, and the characters and props already in the project.",
            Effect = CommandEffect.ReadOnly)]
        internal static async Task<ExitCode> Status(CommandContext cmd)
        {
            var client = new AssetgenClient();
            ServerInfo info = null;
            string error = null;
            try
            {
                info = await client.InfoAsync(cmd.CancellationToken);
            }
            catch (Exception e) when (!(e is OperationCanceledException))
            {
                error = e is AssetgenException ? e.Message : $"no assetgen server at {client.BaseUrl} (start it with: python server.py)";
            }
            cmd.Result.Write(new
            {
                server = client.BaseUrl,
                connected = info != null,
                error,
                version = info?.version,
                devices = info?.devices,
                imageBackend = info?.image_backend?.backend,
                imageBackendReady = info?.image_backend?.ok ?? false,
                busy = info?.busy ?? 0,
                queued = info?.queued ?? 0,
                charactersFolder = AssetgenSettings.instance.charactersFolder,
                characters = CharacterImport.All().Count(),
                propsFolder = AssetgenSettings.instance.propsFolder,
                props = CharacterImport.AllProps().Count(),
                pendingJobs = JobTracker.Pending,
                serverStartedByEditor = AssetgenServer.Owned,
            });
            return ExitCode.Success;
        }

        [CliGroupCommand(Group,
            "Generate an animated character, or with --kind a static prop or diorama, from a description (or a picture). " +
            "Returns the job id at once; the asset is imported when it finishes while the Editor is open. With --wait, " +
            "waits and returns the prefab (several minutes: run with --timeout 1800, or --detach and 'unity job wait').",
            Effect = CommandEffect.Additive, Tags = new[] { "assets/import", "animation" })]
        internal static async Task<ExitCode> Generate(
            CommandContext cmd,
            [CliArg("prompt", "The asset, e.g. \"a goblin shaman with a lantern\". A picture is generated from it unless --image is given.")]
            string prompt = "",
            [CliArg("image", "A picture of the asset (a file path) instead of, or as well as, a description")]
            string image = "",
            [CliArg("kind", "character (rigged and animated), prop (one static object) or diorama (a static set piece)")]
            string kind = "character",
            [CliArg("motions", "Characters' clips as a JSON array, e.g. '[\"An object walks forward.\",\"An object jumps. [once]\"]'. Each loops unless it ends in [once]. Default: the project's motions.")]
            string[] motions = null,
            [CliArg("height", "Height in metres (default: the project's for characters, 1 for props and dioramas)")] float height = 0f,
            [CliArg("name", "Asset name (default: from the description)")] string name = "",
            [CliArg("wait", "Wait for the asset and import it before returning")] bool wait = false,
            [CliArg("idol-steps", "Idol's sampling steps (default: Project Settings > Assetgen)")] int idolSteps = 0,
            [CliArg("motion-steps", "Motion sampler steps (default: Project Settings > Assetgen)")] int motionSteps = 0,
            [CliArg("seed", "Seed (default: Project Settings > Assetgen)")] int seed = -1,
            [CliArg("image-model", "Text-to-image model: an id from 'assetgen image_models', a recipe or a file (default: Project Settings > Assetgen, else automatic)")]
            string imageModel = "")
        {
            if (string.IsNullOrWhiteSpace(prompt) && string.IsNullOrWhiteSpace(image))
            {
                cmd.Log.Error("Give --prompt (a description) or --image (a picture).");
                return ExitCode.InvalidUsage;
            }
            if (!Enum.TryParse(kind, true, out AssetKind assetKind) || !Enum.IsDefined(typeof(AssetKind), assetKind))
            {
                cmd.Log.Error($"--kind is character, prop or diorama, not '{kind}'.");
                return ExitCode.InvalidUsage;
            }
            var settings = AssetgenSettings.instance;
            var request = new JobRequest
            {
                prompt = prompt.Trim(),
                kind = assetKind.Id(),
                name = name.Trim(),
                height = height > 0f ? height : assetKind.DefaultHeight(),
                motions = assetKind.IsStatic() ? Array.Empty<string>()
                    : motions != null && motions.Length > 0 ? motions : settings.defaultMotions.ToArray(),
                settings = settings.generation.ToJob(),
                image_backend = new ImageBackendRequest
                {
                    model = string.IsNullOrEmpty(imageModel) ? settings.generation.imageModel : imageModel,
                },
            };
            if (idolSteps > 0)
                request.settings.idol_steps = idolSteps;
            if (motionSteps > 0)
                request.settings.flow_steps = motionSteps;
            if (seed >= 0)
                request.settings.seed = seed;
            if (!string.IsNullOrWhiteSpace(image))
            {
                if (!File.Exists(image))
                {
                    cmd.Log.Error($"No picture at {image}.");
                    return ExitCode.NotFound;
                }
                request.image = "data:image/png;base64," + Convert.ToBase64String(File.ReadAllBytes(image));
            }

            var client = new AssetgenClient();
            if (!AssetgenServer.Owned || !await Answers(client, cmd))
            {
                cmd.Log.Info("Making sure the assetgen server is running.");
                await AssetgenServer.EnsureRunningAsync(cmd.CancellationToken);
            }
            var job = await client.SubmitAsync(request, cmd.CancellationToken);
            cmd.Log.Info($"Started {job.id}.");
            if (!wait)
            {
                JobTracker.Track(job.id);
                cmd.Result.Write(new
                {
                    job = job.id, status = job.status, server = client.BaseUrl,
                    next = $"unity command assetgen wait --id {job.id}",
                });
                return ExitCode.Success;
            }
            return await Finish(cmd, client, job.id);
        }

        [CliGroupCommand(Group, "The text-to-image models the server's image backend can load (for --image-model).",
            Effect = CommandEffect.ReadOnly)]
        internal static async Task<ExitCode> ImageModels(CommandContext cmd)
        {
            var models = await new AssetgenClient().ImageModelsAsync(cmd.CancellationToken);
            if (!string.IsNullOrEmpty(models.error))
                cmd.Log.Warning(models.error);
            foreach (var o in models.options ?? Array.Empty<ImageModelOption>())
                cmd.Result.WriteRecord(new { o.id, o.label, automatic = o.id == models.automatic,
                                             selected = o.id == AssetgenSettings.instance.generation.imageModel });
            return ExitCode.Success;
        }

        [CliGroupCommand(Group, "Wait for a job to finish, import its asset and return the prefab.",
            Effect = CommandEffect.Additive, Tags = new[] { "assets/import" })]
        internal static async Task<ExitCode> Wait(CommandContext cmd, [CliArg("id", "The job id", Required = true)] string id)
        {
            var imported = JobTracker.ImportedPrefab(id);
            if (!string.IsNullOrEmpty(imported) && File.Exists(imported))
            {
                WriteCharacter(cmd, id, imported);
                return ExitCode.Success;
            }
            return await Finish(cmd, new AssetgenClient(), id);
        }

        [CliGroupCommand(Group, "A job's status and progress, and its prefab once imported.", Effect = CommandEffect.ReadOnly)]
        internal static async Task<ExitCode> Job(CommandContext cmd, [CliArg("id", "The job id", Required = true)] string id)
        {
            JobInfo job;
            try
            {
                job = await new AssetgenClient().JobAsync(id, cmd.CancellationToken);
            }
            catch (AssetgenException e) when (e.Status == 404)
            {
                cmd.Log.Error($"No job {id} on the server.");
                return ExitCode.NotFound;
            }
            cmd.Result.Write(new
            {
                job = job.id, name = job.name, status = job.status, stage = job.stage, progress = job.progress,
                error = job.error, prefab = JobTracker.ImportedPrefab(id),
            });
            return ExitCode.Success;
        }

        [CliGroupCommand(Group, "Stop a queued or running job.", Effect = CommandEffect.Destructive)]
        internal static async Task<ExitCode> Cancel(CommandContext cmd, [CliArg("id", "The job id", Required = true)] string id)
        {
            var job = await new AssetgenClient().CancelAsync(id, cmd.CancellationToken);
            cmd.Result.Write(new { job = job.id, status = job.status });
            return ExitCode.Success;
        }

        [CliGroupCommand(Group,
            "Import a character, prop or diorama: a finished job id, or a path to an export zip, a folder, or a " +
            ".glb/.assetgen pair. Builds its prefab (and a character's Animator Controller) and returns the prefab.",
            Effect = CommandEffect.Additive, Tags = new[] { "assets/import" })]
        internal static async Task<ExitCode> Import(
            CommandContext cmd,
            [CliArg("source", "A job id, or a path on disk", Required = true)] string source)
        {
            if (File.Exists(source) || Directory.Exists(source))
            {
                WriteCharacter(cmd, null, CharacterImport.FromPath(source));
                return ExitCode.Success;
            }
            var prefab = await CharacterImport.FromJobAsync(source, new AssetgenClient(), cmd.CancellationToken);
            WriteCharacter(cmd, source, prefab);
            return ExitCode.Success;
        }

        [CliGroupCommand(Group, "Rebuild a character's prefab and Animator Controller, or a prop's prefab, from its .glb and .assetgen.",
            Effect = CommandEffect.Idempotent)]
        internal static ExitCode Build(
            CommandContext cmd,
            [CliArg("character", "Character or prop name, or the path of its .assetgen file", Required = true)] string character)
        {
            var path = ResolveCharacter(character);
            if (path == null)
            {
                cmd.Log.Error($"No character or prop '{character}' in the project (see: unity command assetgen list).");
                return ExitCode.NotFound;
            }
            var prefab = AssetDatabase.GetAssetPath(CharacterImport.BuildPrefab(path));
            WriteCharacter(cmd, null, prefab);
            return ExitCode.Success;
        }

        [CliGroupCommand(Group, "Every character in the project, with its prefab and clips, and every prop and diorama, with its kind and size.",
            Effect = CommandEffect.ReadOnly)]
        internal static ExitCode List(CommandContext cmd)
        {
            foreach (var (path, definition) in CharacterImport.All())
            {
                var prefab = CharacterBuilder.PrefabPath(path);
                cmd.Result.WriteRecord(new
                {
                    name = definition.name,
                    path,
                    prefab = File.Exists(prefab) ? prefab : null,
                    height = definition.height,
                    clips = definition.clips.Select(c => c.name).ToArray(),
                    locomotion = definition.locomotion.Select(p => p.clip).ToArray(),
                });
            }
            foreach (var (path, definition) in CharacterImport.AllProps())
            {
                var prefab = PropBuilder.PrefabPath(path);
                cmd.Result.WriteRecord(new
                {
                    name = definition.name,
                    kind = definition.kind,
                    path,
                    prefab = File.Exists(prefab) ? prefab : null,
                    height = definition.height,
                    size = new[] { definition.size.x, definition.size.y, definition.size.z },
                    collider = definition.colliderType.ToString().ToLowerInvariant(),
                    footprintRadius = definition.footprintRadius,
                });
            }
            return ExitCode.Success;
        }

        [CliGroupCommand(Group, "Put a character, prop or diorama in the open scene. Returns the new GameObject.",
            Effect = CommandEffect.Additive, Tags = new[] { "gameobjects", "prefabs" })]
        internal static ExitCode Spawn(
            CommandContext cmd,
            [CliArg("character", "Character or prop name, .assetgen path or prefab path", Required = true)] string character,
            [CliArg("position", "World position as x,y,z")] string position = "",
            [CliArg("speed", "Ground speed in m/s: 0 stands, the walk or run speed moves the feet at their pace")] float speed = 0f,
            [CliArg("move-forward", "Move forward at that speed in Play mode")] bool moveForward = false)
        {
            var prefab = CharacterImport.FindPrefab(character);
            if (prefab == null)
            {
                cmd.Log.Error($"No prefab for '{character}' (see: unity command assetgen list).");
                return ExitCode.NotFound;
            }
            Vector3? at = null;
            if (!string.IsNullOrWhiteSpace(position))
            {
                var parts = position.Split(',').Select(p => float.Parse(p.Trim(), CultureInfo.InvariantCulture)).ToArray();
                if (parts.Length != 3)
                    throw new ArgumentException("--position is x,y,z");
                at = new Vector3(parts[0], parts[1], parts[2]);
            }
            var instance = AssetgenWindow.AddToScene(AssetDatabase.GetAssetPath(prefab), at);
            if (instance.TryGetComponent(out AssetgenCharacter animated))
            {
                animated.Speed = speed;
                animated.MoveForward = moveForward;
            }
            EditorSceneManager.MarkSceneDirty(instance.scene);
            var p = instance.transform.position;
            cmd.Result.Write(new
            {
                gameObject = instance.name, instanceId = instance.GetInstanceID(), scene = instance.scene.path,
                position = new[] { p.x, p.y, p.z },
            });
            return ExitCode.Success;
        }

        [CliGroupCommand(Group, "In Play mode: play a clip on a character in the scene, or set its locomotion speed.",
            Effect = CommandEffect.Additive, Tags = new[] { "animation", "runtime" })]
        internal static ExitCode Play(
            CommandContext cmd,
            [CliArg("target", "The character's GameObject name", Required = true)] string target,
            [CliArg("clip", "A clip name, or Locomotion")] string clip = "",
            [CliArg("speed", "Ground speed in m/s for the locomotion blend")] float speed = -1f)
        {
            if (!EditorApplication.isPlaying)
            {
                cmd.Log.Error("Characters animate in Play mode: run 'unity command editor.playmode start' first.");
                return ExitCode.InvalidUsage;
            }
            var found = UnityEngine.Object.FindObjectsByType<AssetgenCharacter>(FindObjectsSortMode.None)
                .FirstOrDefault(c => c.name == target);
            if (found == null)
            {
                cmd.Log.Error($"No AssetgenCharacter called '{target}' in the scene.");
                return ExitCode.NotFound;
            }
            if (speed >= 0f)
                found.Speed = speed;
            if (!string.IsNullOrEmpty(clip) && !found.Play(clip))
            {
                cmd.Log.Error($"{target} has no clip '{clip}'; it has: {string.Join(", ", found.Definition.clips.Select(c => c.name))}.");
                return ExitCode.NotFound;
            }
            cmd.Result.Write(new { target = found.name, clip, speed = found.Speed });
            return ExitCode.Success;
        }

        [CliGroupCommand(ServerGroup,
            "Start the local assetgen server (server.py) and wait until it answers. Uses the folder, Python and " +
            "arguments in Project Settings > Assetgen. Does nothing when a server already answers.",
            Effect = CommandEffect.Idempotent, Name = "start")]
        internal static async Task<ExitCode> ServerStart(CommandContext cmd)
        {
            var client = new AssetgenClient();
            var already = await Answers(client, cmd);
            if (!already)
            {
                var reason = AssetgenServer.CannotStartReason;
                if (reason != null)
                {
                    cmd.Log.Error(reason);
                    return ExitCode.Failed;
                }
                AssetgenServer.Start();
                await AssetgenServer.WaitUntilReadyAsync(client, cmd.CancellationToken);
            }
            WriteServer(cmd, client, await client.InfoAsync(cmd.CancellationToken), already ? "already running" : "started");
            return ExitCode.Success;
        }

        [CliGroupCommand(ServerGroup, "Stop the assetgen server this Editor started, with the stage it is running.",
            Effect = CommandEffect.Destructive, Name = "stop")]
        internal static ExitCode ServerStop(CommandContext cmd)
        {
            if (!AssetgenServer.Owned)
            {
                cmd.Log.Warning("This Editor did not start a server, so there is none to stop.");
                cmd.Result.Write(new { state = "not started by this Editor" });
                return ExitCode.Success;
            }
            AssetgenServer.Stop();
            cmd.Result.Write(new { state = "stopped" });
            return ExitCode.Success;
        }

        [CliGroupCommand(ServerGroup, "Whether the assetgen server answers, and whether this Editor started it.",
            Effect = CommandEffect.ReadOnly, Name = "status")]
        internal static async Task<ExitCode> ServerStatus(CommandContext cmd)
        {
            var client = new AssetgenClient();
            ServerInfo info = null;
            try
            {
                info = await client.InfoAsync(cmd.CancellationToken);
            }
            catch (Exception e) when (!(e is OperationCanceledException) || !cmd.CancellationToken.IsCancellationRequested)
            {
                // reported as not answering
            }
            var state = info != null ? "running" : AssetgenServer.Owned ? "starting" : "stopped";
            WriteServer(cmd, client, info, state);
            return ExitCode.Success;
        }

        [CliGroupCommand(ServerGroup, "The last lines of the server's log (the server Unity started).",
            Effect = CommandEffect.ReadOnly, Name = "log")]
        internal static ExitCode ServerLog(CommandContext cmd, [CliArg("lines", "How many lines")] int lines = 40)
        {
            cmd.Result.Write(new { path = AssetgenServer.LogPath, log = AssetgenServer.LogTail(Math.Max(1, lines)) });
            return ExitCode.Success;
        }

        static void WriteServer(CommandContext cmd, AssetgenClient client, ServerInfo info, string state)
        {
            cmd.Result.Write(new
            {
                state,
                url = client.BaseUrl,
                startedByEditor = AssetgenServer.Owned,
                process = AssetgenServer.Process?.Id,
                version = info?.version,
                devices = info?.devices,
                imageBackend = info?.image_backend?.backend,
                folder = AssetgenServer.ResolvedFolder,
                python = AssetgenServer.ResolvedPython(AssetgenServer.ResolvedFolder),
                log = AssetgenServer.LogPath,
            });
        }

        static async Task<bool> Answers(AssetgenClient client, CommandContext cmd)
        {
            try
            {
                await client.InfoAsync(cmd.CancellationToken);
                return true;
            }
            catch (Exception e) when (!(e is OperationCanceledException) || !cmd.CancellationToken.IsCancellationRequested)
            {
                return false;
            }
        }

        static async Task<ExitCode> Finish(CommandContext cmd, AssetgenClient client, string id)
        {
            var job = await Poll(cmd, client, id);
            if (!job.Done)
            {
                cmd.Log.Error($"{id} {job.status}{(string.IsNullOrEmpty(job.error) ? "" : ": " + job.error)}");
                return job.status == "cancelled" ? ExitCode.Cancelled : ExitCode.Failed;
            }
            var prefab = await CharacterImport.FromJobAsync(id, client, cmd.CancellationToken);
            WriteCharacter(cmd, id, prefab);
            return ExitCode.Success;
        }

        static async Task<JobInfo> Poll(CommandContext cmd, AssetgenClient client, string id)
        {
            string stage = null;
            var next = 0.0;
            while (true)
            {
                cmd.CancellationToken.ThrowIfCancellationRequested();
                if (EditorApplication.timeSinceStartup >= next)
                {
                    var job = await client.JobAsync(id, cmd.CancellationToken);
                    if (!job.Active)
                        return job;
                    if (job.stage != stage)
                    {
                        stage = job.stage;
                        cmd.Log.Info($"{id}: {stage} ({Mathf.RoundToInt(job.progress * 100)}%)");
                    }
                    next = EditorApplication.timeSinceStartup + 2.0;
                }
                await Task.Yield();
            }
        }

        static void WriteCharacter(CommandContext cmd, string job, string prefabPath)
        {
            var definitionPath = Path.ChangeExtension(prefabPath, ".assetgen");
            var prop = AssetDatabase.LoadAssetAtPath<PropDefinition>(definitionPath);
            if (prop != null)
            {
                cmd.Result.Write(new
                {
                    job,
                    prefab = prefabPath,
                    name = prop.name,
                    kind = prop.kind,
                    height = prop.height,
                    size = new[] { prop.size.x, prop.size.y, prop.size.z },
                    collider = prop.colliderType.ToString().ToLowerInvariant(),
                    footprintRadius = prop.footprintRadius,
                    baseRadius = prop.baseRadius,
                    next = $"unity command assetgen spawn --character \"{prefabPath}\"",
                });
                return;
            }
            var definition = AssetDatabase.LoadAssetAtPath<CharacterDefinition>(definitionPath);
            cmd.Result.Write(new
            {
                job,
                prefab = prefabPath,
                controller = CharacterBuilder.ControllerPath(definitionPath),
                name = definition != null ? definition.name : Path.GetFileNameWithoutExtension(prefabPath),
                height = definition != null ? definition.height : 0f,
                clips = definition != null ? definition.clips.Select(c => new { c.name, c.role, c.loop, c.duration, c.speed }).ToArray() : null,
                next = $"unity command assetgen spawn --character \"{prefabPath}\"",
            });
        }

        /// <summary>The .assetgen path of a character, prop or diorama, by name or path.</summary>
        static string ResolveCharacter(string character)
        {
            if (character.EndsWith(".assetgen", StringComparison.OrdinalIgnoreCase) && File.Exists(character))
                return character.Replace('\\', '/');
            return CharacterImport.All().Where(c => c.definition.name == character || c.definition.characterName == character)
                       .Select(c => c.path).FirstOrDefault()
                   ?? CharacterImport.AllProps().Where(p => p.definition.name == character || p.definition.propName == character)
                       .Select(p => p.path).FirstOrDefault();
        }
    }
}
