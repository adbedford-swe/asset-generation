using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Assetgen.Editor
{
    /// <summary>
    /// Turns an imported scene (<c>name.glb</c> + <c>name.assetgen</c>) into <c>name.prefab</c>, a variant of
    /// the model with a MeshCollider on the ground, each placement's collider as its object asks, everything
    /// marked static, and an <see cref="AssetgenScene"/>. Building again updates the prefab in place, so
    /// references to it survive a re-import; colliders added to it by hand are replaced.
    /// </summary>
    public static class SceneBuilder
    {
        public static string PrefabPath(string definitionPath) => Path.ChangeExtension(definitionPath, ".prefab");

        public static string ModelPath(string definitionPath, SceneDefinition definition)
        {
            var folder = Path.GetDirectoryName(definitionPath)?.Replace('\\', '/');
            var file = string.IsNullOrEmpty(definition.modelFile)
                ? Path.GetFileNameWithoutExtension(definitionPath) + ".glb"
                : definition.modelFile;
            return string.IsNullOrEmpty(folder) ? file : $"{folder}/{file}";
        }

        /// <summary>The prop files of the scene's objects (the <c>assets/</c> folder beside it).</summary>
        public static IEnumerable<string> AssetPaths(string definitionPath, SceneDefinition definition)
        {
            var folder = Path.GetDirectoryName(definitionPath)?.Replace('\\', '/');
            foreach (var o in definition.objects)
                if (!string.IsNullOrEmpty(o.asset))
                    yield return string.IsNullOrEmpty(folder) ? o.asset : $"{folder}/{o.asset}";
        }

        /// <summary>Whether the prefab exists and is newer than the files it comes from.</summary>
        public static bool IsUpToDate(string definitionPath)
        {
            var definition = AssetDatabase.LoadAssetAtPath<SceneDefinition>(definitionPath);
            if (definition == null)
                return false;
            var prefab = PrefabPath(definitionPath);
            var sources = new[] { definitionPath, ModelPath(definitionPath, definition) };
            if (!File.Exists(prefab) || sources.Any(p => !File.Exists(p)))
                return false;
            return File.GetLastWriteTimeUtc(prefab) >= sources.Max(File.GetLastWriteTimeUtc);
        }

        /// <summary>Builds (or rebuilds) the scene's prefab and returns it. The objects' props are built on their own.</summary>
        public static GameObject Build(string definitionPath)
        {
            var definition = AssetDatabase.LoadAssetAtPath<SceneDefinition>(definitionPath);
            if (definition == null)
                throw new ArgumentException($"{definitionPath} is not an imported .assetgen scene file");
            var modelPath = ModelPath(definitionPath, definition);
            var model = AssetDatabase.LoadAssetAtPath<GameObject>(modelPath);
            if (model == null)
                throw new InvalidOperationException(
                    $"{modelPath} has not imported as a model. glTF import needs com.unity.cloud.gltfast (a dependency " +
                    "of this package); check the Console for its import errors.");
            var prefab = BuildPrefab(definition, model, PrefabPath(definitionPath));
            AssetDatabase.SaveAssets();
            return prefab;
        }

        static GameObject BuildPrefab(SceneDefinition definition, GameObject model, string path)
        {
            var exists = AssetDatabase.LoadAssetAtPath<GameObject>(path) != null;
            var root = exists ? PrefabUtility.LoadPrefabContents(path) : (GameObject)PrefabUtility.InstantiatePrefab(model);
            try
            {
                PropBuilder.RemoveAddedColliders(root);
                var names = new HashSet<string>(definition.instances.Select(i => i.node)) { definition.groundNode };
                var nodes = Nodes(root.transform, names);

                if (nodes.TryGetValue(definition.groundNode, out var ground))
                    PropBuilder.AddColliders(ground.gameObject, Meshes(ground, nodes), PropDefinition.ColliderType.Mesh,
                        Vector3.zero, Vector3.zero, $"{definition.name}'s ground");
                else
                    Debug.LogWarning($"{definition.name}: the model has no ground node '{definition.groundNode}'");

                var placements = new List<AssetgenScene.Placement>();
                foreach (var instance in definition.instances)
                {
                    var o = definition.Find(instance.objectId);
                    if (!nodes.TryGetValue(instance.node, out var node) || o == null)
                    {
                        Debug.LogWarning($"{definition.name}: skipped the placement '{instance.node}' of '{instance.objectId}': " +
                                         "there is no such node or object");
                        continue;
                    }
                    PropBuilder.AddColliders(node.gameObject, Meshes(node, nodes), o.colliderType, o.boxCenter, o.boxSize,
                        $"{definition.name}/{instance.node}");
                    placements.Add(new AssetgenScene.Placement { objectId = o.id, node = node });
                }

                foreach (var t in root.GetComponentsInChildren<Transform>(true))
                    GameObjectUtility.SetStaticEditorFlags(t.gameObject, PropBuilder.StaticFlags);
                if (!root.TryGetComponent(out AssetgenScene scene))
                    scene = root.AddComponent<AssetgenScene>();
                scene.Definition = definition;
                scene.Ground = ground;
                scene.SetPlacements(placements);
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

        /// <summary>
        /// The scene's nodes by name, the shallowest first. glTFast names each node's GameObject after the node,
        /// under the model's root or a GameObject for the glTF scene, so they are searched for at any depth.
        /// </summary>
        static Dictionary<string, Transform> Nodes(Transform root, HashSet<string> names)
        {
            var nodes = new Dictionary<string, Transform>();
            var queue = new Queue<Transform>();
            foreach (Transform child in root)
                queue.Enqueue(child);
            while (queue.Count > 0)
            {
                var t = queue.Dequeue();
                if (names.Contains(t.name) && !nodes.ContainsKey(t.name))
                    nodes[t.name] = t;
                foreach (Transform child in t)
                    queue.Enqueue(child);
            }
            return nodes;
        }

        /// <summary>
        /// A node's meshes: glTFast puts a node's first primitive on the node's own GameObject and any others on
        /// children, so it is those, without going into other nodes nested under it.
        /// </summary>
        static List<MeshFilter> Meshes(Transform node, Dictionary<string, Transform> nodes)
        {
            var found = new List<MeshFilter>();
            var stack = new Stack<Transform>();
            stack.Push(node);
            while (stack.Count > 0)
            {
                var t = stack.Pop();
                if (t.TryGetComponent(out MeshFilter filter))
                    found.Add(filter);
                foreach (Transform child in t)
                    if (!(nodes.TryGetValue(child.name, out var other) && other == child))
                        stack.Push(child);
            }
            return found;
        }
    }
}
