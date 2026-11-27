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
            [Tooltip("unimate: sampled from the prompt; video: fitted to a video generated from the character's picture")]
            public string source;
            [Tooltip("Moments in the clip (clips made from video): footsteps and a one-shot's impact")]
            public List<ClipEvent> events = new List<ClipEvent>();
            [Tooltip("A one-shot's timing, in seconds (clips made from video)")]
            public Phases phases;
            [Tooltip("The clip's sound, relative to the .assetgen file (clips made from video)")]
            public string sound;

            public bool IsLocomotion => role == "idle" || role == "walk" || role == "run" || role == "move";
        }

        [Serializable]
        public struct ClipEvent
        {
            [Tooltip("Seconds from the clip's start")]
            public float time;
            [Tooltip("contact: part of the body comes to rest on the ground (a footstep); impact: the action lands")]
            public string type;
            [Tooltip("contact: the bone carrying the part on the ground")]
            public string joint;
        }

        /// <summary>A one-shot's wind-up, action and recovery, as (start, end) seconds, and when it lands.</summary>
        [Serializable]
        public struct Phases
        {
            public Vector2 windup;
            public Vector2 active;
            public Vector2 recovery;
            public float impact;

            public bool IsSet => active.y > active.x;

            /// <summary>"windup", "active" or "recovery" at <paramref name="time"/> seconds into the clip; "" without phases.</summary>
            public string At(float time)
            {
                if (!IsSet)
                    return "";
                if (time <= windup.y)
                    return "windup";
                return time <= active.y ? "active" : "recovery";
            }
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
                var clip = new Clip
                {
                    name = c.name, prompt = c.prompt, role = c.role, loop = c.loop, duration = c.duration,
                    speed = c.speed, rootVelocity = v, source = string.IsNullOrEmpty(c.source) ? "unimate" : c.source,
                    sound = c.sound, phases = c.phases?.ToPhases() ?? default
                };
                foreach (var e in c.events ?? Array.Empty<EventData>())
                    clip.events.Add(new ClipEvent { time = e.time, type = e.type, joint = e.joint });
                definition.clips.Add(clip);
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
            public string source;
            public EventData[] events;
            public PhasesData phases;
            public string sound;
        }

        [Serializable] class EventData { public float time; public string type; public string joint; }

        [Serializable] class PhasesData
        {
            public float[] windup;
            public float[] active;
            public float[] recovery;
            public float impact;

            static Vector2 Span(float[] s) => s != null && s.Length == 2 ? new Vector2(s[0], s[1]) : Vector2.zero;

            public Phases ToPhases() => new Phases
            {
                windup = Span(windup), active = Span(active), recovery = Span(recovery), impact = impact
            };
        }
#pragma warning restore 0649
    }
}
