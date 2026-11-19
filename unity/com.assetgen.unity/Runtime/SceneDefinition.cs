using System;
using System.Collections.Generic;
using UnityEngine;

namespace Assetgen
{
    /// <summary>
    /// An assetgen scene file (<c>.assetgen</c>, format <c>assetgen.scene</c>) as an asset: the ground, the
    /// kinds of object in the scene and every placement of them. The importer makes one from the file; the
    /// prefab's <see cref="AssetgenScene"/> holds it, and maps each placement's GameObject to its object.
    /// </summary>
    /// <remarks>
    /// Positions and rotations are in Unity axes: the file's are glTF's, whose x is negated on import, as
    /// glTFast does to the model itself. Each object is also imported on its own, as a prop, from the
    /// files its <see cref="SceneObject.asset"/> and <see cref="SceneObject.file"/> name.
    /// </remarks>
    public sealed class SceneDefinition : ScriptableObject
    {
        public const string Format = "assetgen.scene";

        [Serializable]
        public sealed class SceneObject
        {
            public string id;
            [Tooltip("prop or diorama")]
            public string kind;
            [TextArea] public string prompt;
            [Tooltip("The object's model on its own, relative to the scene file")]
            public string file;
            [Tooltip("Its prop file, relative to the scene file")]
            public string asset;
            [Tooltip("Height in metres, before each placement's scale")]
            public float height;
            public PropDefinition.ColliderType colliderType;
            [Tooltip("The collider box at the object's own scale")]
            public Vector3 boxCenter;
            public Vector3 boxSize;
            public float footprintRadius;
            public float baseRadius;
            [Tooltip("How many times it is placed")]
            public int count;

            public bool IsDiorama => kind == PropDefinition.DioramaKind;
        }

        [Serializable]
        public sealed class Instance
        {
            public string objectId;
            [Tooltip("The placement's node in the model: <object>_<k>")]
            public string node;
            public Vector3 position;
            public Quaternion rotation = Quaternion.identity;
            [Tooltip("Turn about the up axis in degrees, as Unity measures it")]
            public float yaw;
            public float scale = 1f;
            [Tooltip("Placed where the spec says, rather than scattered")]
            public bool placed;
        }

        public string sceneName;
        public string style;
        public string modelFile;
        [Tooltip("The ground's width (x) and depth (z) in metres, centred on the origin")]
        public Vector2 size;
        [Tooltip("The ground's node in the model")]
        public string groundNode = "ground";
        [Tooltip("The ground's lowest and highest points, in metres")]
        public Vector2 heightRange;
        public List<SceneObject> objects = new List<SceneObject>();
        public List<Instance> instances = new List<Instance>();
        public List<string> warnings = new List<string>();

        public SceneObject Find(string objectId) => objects.Find(o => o.id == objectId);

        /// <summary>Reads a scene file. Throws <see cref="FormatException"/> on anything else.</summary>
        public static SceneDefinition FromJson(string json)
        {
            var file = JsonUtility.FromJson<FileData>(json);
            if (file == null || file.format != Format)
                throw new FormatException($"not an assetgen scene file (format is not \"{Format}\")");
            if (file.version > 1)
                throw new FormatException($"scene format v{file.version} is newer than this package reads (v1); update com.assetgen.unity");

            var definition = CreateInstance<SceneDefinition>();
            definition.sceneName = file.name;
            definition.style = file.style;
            definition.modelFile = file.model?.file;
            definition.size = Vector2Of(file.model?.size ?? file.ground?.size);
            if (!string.IsNullOrEmpty(file.ground?.node))
                definition.groundNode = file.ground.node;
            definition.heightRange = Vector2Of(file.ground?.height_range);
            foreach (var o in file.objects ?? Array.Empty<ObjectData>())
            {
                definition.objects.Add(new SceneObject
                {
                    id = o.id, kind = string.IsNullOrEmpty(o.kind) ? PropDefinition.PropKind : o.kind, prompt = o.prompt,
                    file = o.file, asset = o.asset, height = o.height,
                    colliderType = PropDefinition.ParseCollider(o.collider),
                    boxCenter = PropDefinition.Position(o.box?.center), boxSize = PropDefinition.Size(o.box?.size),
                    footprintRadius = o.footprint?.radius ?? 0f, baseRadius = o.footprint?.base_radius ?? 0f,
                    count = o.count,
                });
            }
            foreach (var i in file.instances ?? Array.Empty<InstanceData>())
            {
                var q = i.rotation != null && i.rotation.Length == 4
                    ? new Quaternion(i.rotation[0], -i.rotation[1], -i.rotation[2], i.rotation[3]) // mirrored in x, as glTFast does
                    : Quaternion.identity;
                definition.instances.Add(new Instance
                {
                    objectId = i.@object, node = i.node, position = PropDefinition.Position(i.position), rotation = q,
                    yaw = Mathf.Repeat(-i.yaw, 360f), scale = i.scale > 0f ? i.scale : 1f, placed = i.placed,
                });
            }
            definition.warnings.AddRange(file.warnings ?? Array.Empty<string>());
            return definition;
        }

        static Vector2 Vector2Of(float[] v) => v != null && v.Length == 2 ? new Vector2(v[0], v[1]) : Vector2.zero;

        // The JSON layout, for JsonUtility. Field names follow the file.
#pragma warning disable 0649
        [Serializable] class FileData
        {
            public string format;
            public int version;
            public string name;
            public string style;
            public ModelData model;
            public GroundData ground;
            public ObjectData[] objects;
            public InstanceData[] instances;
            public string[] warnings;
        }

        [Serializable] class ModelData { public string file; public float[] size; }
        [Serializable] class GroundData { public string node; public float[] size; public float[] height_range; }
        [Serializable] class BoxData { public float[] center; public float[] size; }
        [Serializable] class FootprintData { public float radius; public float base_radius; }

        [Serializable] class ObjectData
        {
            public string id;
            public string kind;
            public string prompt;
            public string file;
            public string asset;
            public float height;
            public string collider;
            public BoxData box;
            public FootprintData footprint;
            public int count;
        }

        [Serializable] class InstanceData
        {
            public string @object;
            public string node;
            public float[] position;
            public float[] rotation;
            public float yaw;
            public float scale;
            public bool placed;
        }
#pragma warning restore 0649
    }
}
