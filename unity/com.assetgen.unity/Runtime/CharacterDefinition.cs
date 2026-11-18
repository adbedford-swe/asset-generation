using System;
using System.Collections.Generic;
using UnityEngine;

namespace Assetgen
{
    /// <summary>
    /// An assetgen character file (<c>.assetgen</c>) as an asset: what each clip is for, whether it
    /// loops and how fast it travels. The importer makes one from the file; the prefab's
    /// <see cref="AssetgenCharacter"/> reads it at runtime.
    /// </summary>
    public sealed class CharacterDefinition : ScriptableObject
    {
        /// <summary>The Animator state that blends the locomotion clips by <see cref="SpeedParameter"/>.</summary>
        public const string LocomotionState = "Locomotion";

        /// <summary>The Animator float parameter, in metres per second, that drives the locomotion blend.</summary>
        public const string SpeedParameter = "Speed";

        [Serializable]
        public sealed class Clip
        {
            public string name;
            public string prompt;
            [Tooltip("idle, walk, run or move (locomotion); loop or action otherwise")]
            public string role;
            public bool loop;
            public float duration;
            [Tooltip("Ground speed the clip was made at, in metres per second (looping clips)")]
            public float speed;
            [Tooltip("Velocity the clip travels at, in Unity axes; the clip itself plays in place")]
            public Vector3 rootVelocity;

            public bool IsLocomotion => role == "idle" || role == "walk" || role == "run" || role == "move";
        }

        [Serializable]
        public struct BlendPoint
        {
            public string clip;
            public float speed;
        }

        public string characterName;
        [TextArea] public string prompt;
        public string modelFile;
        public float height;
        public int jointCount;
        public string idleClip;
        public List<Clip> clips = new List<Clip>();
        [Tooltip("Clips blended by ground speed, slowest first")]
        public List<BlendPoint> locomotion = new List<BlendPoint>();
        public string license;

        public Clip Find(string clipName) => clips.Find(c => c.name == clipName);

        /// <summary>The fastest locomotion clip's speed, or 0 when there is no locomotion blend.</summary>
        public float MaxLocomotionSpeed => locomotion.Count > 0 ? locomotion[locomotion.Count - 1].speed : 0f;

        /// <summary>Reads a character file. Throws <see cref="FormatException"/> on anything else.</summary>
        public static CharacterDefinition FromJson(string json)
        {
            var file = JsonUtility.FromJson<FileData>(json);
            if (file == null || file.format != "assetgen.character")
                throw new FormatException("not an assetgen character file (format is not \"assetgen.character\")");
            if (file.version > 1)
                throw new FormatException($"character format v{file.version} is newer than this package reads (v1); update com.assetgen.unity");

            var definition = CreateInstance<CharacterDefinition>();
            definition.characterName = file.name;
            definition.prompt = file.source?.prompt;
            definition.modelFile = file.model?.file;
            definition.height = file.model?.height ?? 0f;
            definition.jointCount = file.skeleton?.joints?.Length ?? 0;
            definition.idleClip = file.locomotion?.idle;
            definition.license = file.license?.note;
            foreach (var c in file.clips ?? Array.Empty<ClipData>())
            {
                var v = c.root_velocity != null && c.root_velocity.Length == 3
                    ? new Vector3(-c.root_velocity[0], c.root_velocity[1], c.root_velocity[2]) // glTF is right-handed
                    : Vector3.zero;
                definition.clips.Add(new Clip
                {
                    name = c.name, prompt = c.prompt, role = c.role, loop = c.loop, duration = c.duration,
                    speed = c.speed, rootVelocity = v
                });
            }
            foreach (var b in file.locomotion?.blend ?? Array.Empty<BlendData>())
                definition.locomotion.Add(new BlendPoint { clip = b.clip, speed = b.speed });
            return definition;
        }

        // The JSON layout, for JsonUtility. Field names follow the file.
#pragma warning disable 0649
        [Serializable] class FileData
        {
            public string format;
            public int version;
            public string name;
            public ModelData model;
            public SkeletonData skeleton;
            public ClipData[] clips;
            public LocomotionData locomotion;
            public SourceData source;
            public LicenseData license;
        }

        [Serializable] class ModelData { public string file; public float height; }
        [Serializable] class SkeletonData { public JointData[] joints; }
        [Serializable] class JointData { public string name; }
        [Serializable] class SourceData { public string prompt; }
        [Serializable] class LicenseData { public string note; }
        [Serializable] class LocomotionData { public string idle; public BlendData[] blend; }
        [Serializable] class BlendData { public string clip; public float speed; }

        [Serializable] class ClipData
        {
            public string name;
            public string prompt;
            public string role;
            public bool loop;
            public float duration;
            public float speed;
            public float[] root_velocity;
        }
#pragma warning restore 0649
    }
}
