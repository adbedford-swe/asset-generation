using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Assetgen.Editor
{
    /// <summary>
    /// Turns an imported character (<c>name.glb</c> + <c>name.assetgen</c>) into something to drop in
    /// a scene: <c>name.controller</c>, an Animator Controller holding a copy of every clip (looping
    /// or not, as the character file says) with a "Locomotion" blend tree on Speed and a state per
    /// clip; and <c>name.prefab</c>, a variant of the model with that controller and an
    /// <see cref="AssetgenCharacter"/>. Building again updates both in place, so references to them
    /// survive a re-import; changes made by hand to the controller are replaced.
    /// <para>
    /// Clips made from video get animation events for their footsteps and impact, and one at their start
    /// for their sound; the prefab then has an AudioSource and the sounds (<c>sounds/</c> beside the
    /// .assetgen file) on its <see cref="AssetgenCharacter"/>.
    /// </para>
    /// </summary>
    public static class CharacterBuilder
    {
        public static string ControllerPath(string definitionPath) => Path.ChangeExtension(definitionPath, ".controller");

        public static string PrefabPath(string definitionPath) => Path.ChangeExtension(definitionPath, ".prefab");

        public static string ModelPath(string definitionPath, CharacterDefinition definition)
        {
            var folder = Path.GetDirectoryName(definitionPath)?.Replace('\\', '/');
            var file = string.IsNullOrEmpty(definition.modelFile)
                ? Path.GetFileNameWithoutExtension(definitionPath) + ".glb"
                : definition.modelFile;
            return string.IsNullOrEmpty(folder) ? file : $"{folder}/{file}";
        }

        /// <summary>Whether the controller and prefab exist and are newer than the files they come from.</summary>
        public static bool IsUpToDate(string definitionPath)
        {
            var definition = AssetDatabase.LoadAssetAtPath<CharacterDefinition>(definitionPath);
            if (definition == null)
                return false;
            var made = new[] { ControllerPath(definitionPath), PrefabPath(definitionPath) };
            var sources = new[] { definitionPath, ModelPath(definitionPath, definition) };
            if (made.Any(p => !File.Exists(p)) || sources.Any(p => !File.Exists(p)))
                return false;
            return made.Min(File.GetLastWriteTimeUtc) >= sources.Max(File.GetLastWriteTimeUtc);
        }

        /// <summary>Builds (or rebuilds) the controller and prefab; returns the prefab.</summary>
        public static GameObject Build(string definitionPath)
        {
            var definition = AssetDatabase.LoadAssetAtPath<CharacterDefinition>(definitionPath);
            if (definition == null)
                throw new ArgumentException($"{definitionPath} is not an imported .assetgen character file");
            var modelPath = ModelPath(definitionPath, definition);
            var model = AssetDatabase.LoadAssetAtPath<GameObject>(modelPath);
            if (model == null)
                throw new InvalidOperationException(
                    $"{modelPath} has not imported as a model. glTF import needs com.unity.cloud.gltfast (a dependency " +
                    "of this package); check the Console for its import errors.");
            var sources = AssetDatabase.LoadAllAssetsAtPath(modelPath).OfType<AnimationClip>()
                .Where(c => !c.name.StartsWith("__preview__", StringComparison.Ordinal)).ToList();
            if (sources.Count == 0)
                Debug.LogWarning($"{modelPath} has no animation clips; was it imported with animation turned off?");

            var controller = BuildController(definition, sources, ControllerPath(definitionPath));
            var prefab = BuildPrefab(definition, model, controller, Sounds(definitionPath, definition),
                                     PrefabPath(definitionPath));
            AssetDatabase.SaveAssets();
            return prefab;
        }

        static AnimatorController BuildController(CharacterDefinition definition, List<AnimationClip> sources, string path)
        {
            var controller = AssetDatabase.LoadAssetAtPath<AnimatorController>(path);
            if (controller == null)
            {
                controller = AnimatorController.CreateAnimatorControllerAtPath(path);
            }
            else
            {
                controller.layers = Array.Empty<AnimatorControllerLayer>();
                controller.parameters = Array.Empty<AnimatorControllerParameter>();
                foreach (var sub in AssetDatabase.LoadAllAssetsAtPath(path))
                    if (sub != null && sub != controller)
                        Object.DestroyImmediate(sub, true);
                controller.AddLayer("Base Layer");
            }
            controller.AddParameter(CharacterDefinition.SpeedParameter, AnimatorControllerParameterType.Float);

            var clips = new Dictionary<string, AnimationClip>();
            for (var i = 0; i < definition.clips.Count; i++)
            {
                var info = definition.clips[i];
                var source = sources.FirstOrDefault(c => c.name == info.name)
                             ?? (sources.Count == definition.clips.Count ? sources[i] : null);
                if (source == null)
                {
                    Debug.LogWarning($"{definition.name}: the model has no clip named '{info.name}'");
                    continue;
                }
                var clip = Object.Instantiate(source);
                clip.name = info.name;
                var settings = AnimationUtility.GetAnimationClipSettings(clip);
                settings.loopTime = info.loop;
                AnimationUtility.SetAnimationClipSettings(clip, settings);
                AnimationUtility.SetAnimationEvents(clip, Events(info, clip.length));
                AssetDatabase.AddObjectToAsset(clip, controller);
                clips[info.name] = clip;
            }

            var machine = controller.layers[0].stateMachine;
            AnimatorState locomotion;
            var blend = definition.locomotion.Where(p => clips.ContainsKey(p.clip)).ToList();
            if (blend.Count >= 2)
            {
                locomotion = controller.CreateBlendTreeInController(CharacterDefinition.LocomotionState, out var tree, 0);
                tree.blendType = BlendTreeType.Simple1D;
                tree.blendParameter = CharacterDefinition.SpeedParameter;
                tree.useAutomaticThresholds = false;
                foreach (var point in blend)
                    tree.AddChild(clips[point.clip], point.speed);
            }
            else
            {
                locomotion = machine.AddState(CharacterDefinition.LocomotionState);
                locomotion.motion = IdleMotion(definition, clips);
            }
            machine.defaultState = locomotion;

            var row = 0;
            foreach (var info in definition.clips)
            {
                if (!clips.TryGetValue(info.name, out var clip) || info.name == CharacterDefinition.LocomotionState)
                    continue;
                var state = machine.AddState(info.name, new Vector3(520f, 60f * row++, 0f));
                state.motion = clip;
                if (info.loop)
                    continue;
                // A one-shot goes back to locomotion as it ends, fading over its last 0.15 s.
                var back = state.AddTransition(locomotion);
                back.hasExitTime = true;
                back.hasFixedDuration = true;
                back.duration = 0.15f;
                back.exitTime = info.duration > 0.3f ? 1f - 0.15f / info.duration : 0.5f;
            }
            EditorUtility.SetDirty(controller);
            return controller;
        }

        /// <summary>The animation events for a clip's timing; <see cref="AssetgenCharacter"/> receives them.</summary>
        public static AnimationEvent[] Events(CharacterDefinition.Clip info, float length)
        {
            var events = new List<AnimationEvent>();
            if (!string.IsNullOrEmpty(info.sound))
                events.Add(new AnimationEvent { time = 0f, functionName = "AssetgenSound", stringParameter = info.name });
            foreach (var e in info.events)
            {
                var time = Mathf.Clamp(e.time, 0f, length);
                if (e.type == "contact")
                    events.Add(new AnimationEvent
                    {
                        time = time, functionName = "AssetgenContact", stringParameter = $"{info.name}|{e.joint}"
                    });
                else if (e.type == "impact")
                    events.Add(new AnimationEvent { time = time, functionName = "AssetgenImpact", stringParameter = info.name });
            }
            return events.OrderBy(e => e.time).ToArray();
        }

        static List<AssetgenCharacter.ClipSound> Sounds(string definitionPath, CharacterDefinition definition)
        {
            var folder = Path.GetDirectoryName(definitionPath)?.Replace('\\', '/');
            var sounds = new List<AssetgenCharacter.ClipSound>();
            foreach (var info in definition.clips)
            {
                if (string.IsNullOrEmpty(info.sound))
                    continue;
                var path = string.IsNullOrEmpty(folder) ? info.sound : $"{folder}/{info.sound}";
                var sound = AssetDatabase.LoadAssetAtPath<AudioClip>(path);
                if (sound == null)
                    Debug.LogWarning($"{definition.name}: no sound at {path} for '{info.name}'");
                else
                    sounds.Add(new AssetgenCharacter.ClipSound { clip = info.name, sound = sound });
            }
            return sounds;
        }

        static Motion IdleMotion(CharacterDefinition definition, Dictionary<string, AnimationClip> clips)
        {
            if (!string.IsNullOrEmpty(definition.idleClip) && clips.TryGetValue(definition.idleClip, out var idle))
                return idle;
            var firstLoop = definition.clips.FirstOrDefault(c => c.loop && clips.ContainsKey(c.name));
            if (firstLoop != null)
                return clips[firstLoop.name];
            return clips.Values.FirstOrDefault();
        }

        static GameObject BuildPrefab(CharacterDefinition definition, GameObject model, AnimatorController controller,
                                      List<AssetgenCharacter.ClipSound> sounds, string path)
        {
            var exists = AssetDatabase.LoadAssetAtPath<GameObject>(path) != null;
            var root = exists ? PrefabUtility.LoadPrefabContents(path) : (GameObject)PrefabUtility.InstantiatePrefab(model);
            try
            {
                if (!root.TryGetComponent(out Animator animator))
                    animator = root.AddComponent<Animator>();
                animator.runtimeAnimatorController = controller;
                animator.applyRootMotion = false;
                if (!root.TryGetComponent(out AssetgenCharacter character))
                    character = root.AddComponent<AssetgenCharacter>();
                character.Definition = definition;
                character.Sounds.Clear();
                character.Sounds.AddRange(sounds);
                if (sounds.Count > 0 && !root.TryGetComponent(out AudioSource audio))
                {
                    audio = root.AddComponent<AudioSource>();
                    audio.playOnAwake = false;
                    audio.spatialBlend = 1f;
                }
                var prefab = PrefabUtility.SaveAsPrefabAsset(root, path, out var saved);
                if (!saved)
                    throw new InvalidOperationException($"could not save {path}");
                return prefab;
            }
            finally
            {
                if (exists)
                    PrefabUtility.UnloadPrefabContents(root);
                else
                    Object.DestroyImmediate(root);
            }
        }
    }

    /// <summary>Builds a character, prop, diorama or scene once both its model and its .assetgen file have imported.</summary>
    sealed class CharacterPostprocessor : AssetPostprocessor
    {
        static readonly HashSet<string> s_Pending = new HashSet<string>();

        static void OnPostprocessAllAssets(string[] imported, string[] deleted, string[] moved, string[] movedFrom)
        {
            if (!AssetgenSettings.instance.buildOnImport)
                return;
            foreach (var path in imported)
            {
                if (path.EndsWith(".assetgen", StringComparison.OrdinalIgnoreCase))
                    s_Pending.Add(path);
                else if (path.EndsWith(".glb", StringComparison.OrdinalIgnoreCase) ||
                         path.EndsWith(".gltf", StringComparison.OrdinalIgnoreCase))
                {
                    var definition = Path.ChangeExtension(path, ".assetgen");
                    if (File.Exists(definition))
                        s_Pending.Add(definition);
                }
            }
            if (s_Pending.Count > 0)
            {
                EditorApplication.delayCall -= BuildPending;
                EditorApplication.delayCall += BuildPending;
            }
        }

        static void BuildPending()
        {
            var paths = s_Pending.ToList();
            s_Pending.Clear();
            foreach (var path in paths)
            {
                var model = CharacterImport.ModelPathOf(path);
                if (model == null || CharacterImport.IsUpToDate(path))
                    continue;
                if (AssetDatabase.LoadAssetAtPath<GameObject>(model) == null)
                    continue; // the model's own import brings us back here
                try
                {
                    Debug.Log($"Assetgen: built {AssetDatabase.GetAssetPath(CharacterImport.BuildPrefab(path))}");
                }
                catch (Exception e)
                {
                    Debug.LogError($"Assetgen: could not build {path}: {e.Message}");
                }
            }
        }
    }
}
