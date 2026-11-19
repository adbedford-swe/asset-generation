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
    /// Turns an imported prop or diorama (<c>name.glb</c> + <c>name.assetgen</c>) into <c>name.prefab</c>,
    /// a variant of the model with the collider its file asks for and an <see cref="AssetgenProp"/>.
    /// Dioramas are marked static, since they never move. Building again updates the prefab in place,
    /// so references to it survive a re-import; colliders added to it by hand are replaced.
    /// </summary>
    public static class PropBuilder
    {
        /// <summary>What dioramas and scenes are marked: everything static, as they never move.</summary>
        internal const StaticEditorFlags StaticFlags = StaticEditorFlags.ContributeGI | StaticEditorFlags.OccluderStatic |
                                               StaticEditorFlags.OccludeeStatic | StaticEditorFlags.BatchingStatic |
                                               StaticEditorFlags.ReflectionProbeStatic;

        public static string PrefabPath(string definitionPath) => Path.ChangeExtension(definitionPath, ".prefab");

        public static string ModelPath(string definitionPath, PropDefinition definition)
        {
            var folder = Path.GetDirectoryName(definitionPath)?.Replace('\\', '/');
            var file = string.IsNullOrEmpty(definition.modelFile)
                ? Path.GetFileNameWithoutExtension(definitionPath) + ".glb"
                : definition.modelFile;
            return string.IsNullOrEmpty(folder) ? file : $"{folder}/{file}";
        }

        /// <summary>Whether the prefab exists and is newer than the files it comes from.</summary>
        public static bool IsUpToDate(string definitionPath)
        {
            var definition = AssetDatabase.LoadAssetAtPath<PropDefinition>(definitionPath);
            if (definition == null)
                return false;
            var prefab = PrefabPath(definitionPath);
            var sources = new[] { definitionPath, ModelPath(definitionPath, definition) };
            if (!File.Exists(prefab) || sources.Any(p => !File.Exists(p)))
                return false;
            return File.GetLastWriteTimeUtc(prefab) >= sources.Max(File.GetLastWriteTimeUtc);
        }

        /// <summary>Builds (or rebuilds) the prefab and returns it.</summary>
        public static GameObject Build(string definitionPath)
        {
            var definition = AssetDatabase.LoadAssetAtPath<PropDefinition>(definitionPath);
            if (definition == null)
                throw new ArgumentException($"{definitionPath} is not an imported .assetgen prop file");
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

        static GameObject BuildPrefab(PropDefinition definition, GameObject model, string path)
        {
            var exists = AssetDatabase.LoadAssetAtPath<GameObject>(path) != null;
            var root = exists ? PrefabUtility.LoadPrefabContents(path) : (GameObject)PrefabUtility.InstantiatePrefab(model);
            try
            {
                RemoveAddedColliders(root);
                var meshes = root.GetComponentsInChildren<MeshFilter>(true).ToList();
                AddColliders(root, meshes, definition.colliderType, definition.boxCenter, definition.boxSize, definition.name);
                var flags = definition.IsDiorama ? StaticFlags : (StaticEditorFlags)0;
                foreach (var t in root.GetComponentsInChildren<Transform>(true))
                    GameObjectUtility.SetStaticEditorFlags(t.gameObject, flags);
                if (!root.TryGetComponent(out AssetgenProp prop))
                    prop = root.AddComponent<AssetgenProp>();
                prop.Definition = definition;
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

        /// <summary>Removes the colliders an earlier build added, so a rebuild starts from the model's own.</summary>
        internal static void RemoveAddedColliders(GameObject root)
        {
            foreach (var collider in root.GetComponentsInChildren<Collider>(true))
                if (PrefabUtility.IsAddedComponentOverride(collider))
                    Object.DestroyImmediate(collider);
        }

        /// <summary>
        /// A collider of <paramref name="type"/>: a BoxCollider on <paramref name="target"/>, or a MeshCollider
        /// beside each of <paramref name="meshes"/> (convex or not).
        /// </summary>
        internal static void AddColliders(GameObject target, IEnumerable<MeshFilter> meshes, PropDefinition.ColliderType type,
                                          Vector3 boxCenter, Vector3 boxSize, string what)
        {
            switch (type)
            {
                case PropDefinition.ColliderType.Box:
                {
                    var box = target.AddComponent<BoxCollider>();
                    box.center = boxCenter;
                    box.size = boxSize;
                    break;
                }
                case PropDefinition.ColliderType.Convex:
                case PropDefinition.ColliderType.Mesh:
                {
                    var filters = meshes.Where(f => f.sharedMesh != null).ToList();
                    if (filters.Count == 0)
                        Debug.LogWarning($"{what}: the model has no mesh to build a collider from");
                    foreach (var filter in filters)
                    {
                        var collider = filter.gameObject.AddComponent<MeshCollider>();
                        collider.sharedMesh = filter.sharedMesh;
                        collider.convex = type == PropDefinition.ColliderType.Convex;
                    }
                    break;
                }
            }
        }
    }
}
