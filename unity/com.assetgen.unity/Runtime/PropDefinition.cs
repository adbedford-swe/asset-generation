using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;
using UnityEngine;

namespace Assetgen
{
    /// <summary>
    /// An assetgen prop file (<c>.assetgen</c>, format <c>assetgen.prop</c>) as an asset: the size,
    /// collider and footprint of a prop or a diorama. The importer makes one from the file; the
    /// prefab's <see cref="AssetgenProp"/> holds it for game code and scene layouts.
    /// </summary>
    /// <remarks>
    /// The model's origin is the centre of its base, on the ground. Positions here are in Unity axes:
    /// the file's are glTF's, whose x is negated on import, as glTFast does to the model itself.
    /// </remarks>
    public sealed class PropDefinition : ScriptableObject
    {
        public const string Format = "assetgen.prop";
        public const string PropKind = "prop";
        public const string DioramaKind = "diorama";

        /// <summary>How the prefab collides: each mesh's convex hull, the meshes themselves, or the box.</summary>
        public enum ColliderType
        {
            Convex,
            Mesh,
            Box,
            None,
        }

        public string propName;
        [Tooltip("prop (one object) or diorama (a whole set piece)")]
        public string kind = PropKind;
        [TextArea] public string prompt;
        public string modelFile;
        [Tooltip("Height in metres")]
        public float height;
        [Tooltip("Width (x), height (y) and depth (z) in metres")]
        public Vector3 size;
        public ColliderType colliderType;
        public Vector3 boxCenter;
        public Vector3 boxSize;
        [Tooltip("The farthest the asset reaches from its origin, seen from above, in metres")]
        public float footprintRadius;
        [Tooltip("The same for its base (the bottom tenth of its height): a tree's trunk rather than its canopy")]
        public float baseRadius;
        [Tooltip("The outline seen from above, as (x, z) points, counter-clockwise")]
        public List<Vector2> outline = new List<Vector2>();

        public bool IsDiorama => kind == DioramaKind;

        /// <summary>Reads a prop file. Throws <see cref="FormatException"/> on anything else.</summary>
        public static PropDefinition FromJson(string json)
        {
            var file = JsonUtility.FromJson<FileData>(json);
            if (file == null || file.format != Format)
                throw new FormatException($"not an assetgen prop file (format is not \"{Format}\")");
            if (file.version > 1)
                throw new FormatException($"prop format v{file.version} is newer than this package reads (v1); update com.assetgen.unity");

            var definition = CreateInstance<PropDefinition>();
            definition.propName = file.name;
            definition.kind = string.IsNullOrEmpty(file.kind) ? PropKind : file.kind;
            definition.prompt = file.source?.prompt;
            definition.modelFile = file.model?.file;
            definition.height = file.model?.height ?? 0f;
            definition.size = Size(file.model?.size);
            definition.colliderType = ParseCollider(file.collider?.type);
            definition.boxCenter = Position(file.collider?.box?.center);
            definition.boxSize = Size(file.collider?.box?.size);
            definition.footprintRadius = file.footprint?.radius ?? 0f;
            definition.baseRadius = file.footprint?.base_radius ?? 0f;
            definition.outline = Outline(json);
            return definition;
        }

        // Every file has the box, so a collider type this package does not know falls back to it.
        static ColliderType ParseCollider(string type)
        {
            switch (type)
            {
                case "convex": return ColliderType.Convex;
                case "mesh": return ColliderType.Mesh;
                case "none": return ColliderType.None;
                default: return ColliderType.Box;
            }
        }

        static Vector3 Size(float[] v) => v != null && v.Length == 3 ? new Vector3(v[0], v[1], v[2]) : Vector3.zero;

        static Vector3 Position(float[] v) => v != null && v.Length == 3 ? new Vector3(-v[0], v[1], v[2]) : Vector3.zero;

        static readonly Regex OutlinePattern = new Regex(@"""outline""\s*:\s*\[((?:\s*\[[^\[\]]*\]\s*,?)*)\s*\]");
        static readonly Regex PointPattern = new Regex(@"\[\s*([^,\[\]]+?)\s*,\s*([^,\[\]]+?)\s*\]");

        // JsonUtility cannot read an array of arrays, so the outline's [x, z] pairs are read from the text.
        // Mirroring x reverses the outline's turn, so it is walked backwards to stay counter-clockwise.
        static List<Vector2> Outline(string json)
        {
            var points = new List<Vector2>();
            var match = OutlinePattern.Match(json);
            if (!match.Success)
                return points;
            foreach (Match point in PointPattern.Matches(match.Groups[1].Value))
            {
                if (float.TryParse(point.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) &&
                    float.TryParse(point.Groups[2].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var z))
                    points.Add(new Vector2(-x, z));
            }
            points.Reverse();
            return points;
        }

        // The JSON layout, for JsonUtility. Field names follow the file.
#pragma warning disable 0649
        [Serializable] class FileData
        {
            public string format;
            public int version;
            public string name;
            public string kind;
            public ModelData model;
            public ColliderData collider;
            public FootprintData footprint;
            public SourceData source;
        }

        [Serializable] class ModelData { public string file; public float height; public float[] size; }
        [Serializable] class ColliderData { public string type; public BoxData box; }
        [Serializable] class BoxData { public float[] center; public float[] size; }
        [Serializable] class FootprintData { public float radius; public float base_radius; }
        [Serializable] class SourceData { public string prompt; }
#pragma warning restore 0649
    }
}
