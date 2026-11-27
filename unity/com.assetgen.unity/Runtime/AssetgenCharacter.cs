using System;
using System.Collections.Generic;
using UnityEngine;

namespace Assetgen
{
    /// <summary>
    /// Plays an assetgen character. Set <see cref="Speed"/> (metres per second) and the locomotion
    /// blend picks idle, walk or run at the pace the clips were made for; call <see cref="Play"/>
    /// for anything else. One-shot clips return to locomotion when they end.
    /// <para>
    /// Clips made from video carry their timing as animation events: <see cref="Footstep"/> when part of
    /// the body comes to rest on the ground, <see cref="Impact"/> when a one-shot's action lands (hit
    /// boxes, damage, camera shake), and <see cref="ClipEvent"/> for both. Their sounds play from the
    /// prefab's AudioSource as the clip starts.
    /// </para>
    /// </summary>
    [RequireComponent(typeof(Animator))]
    [DisallowMultipleComponent]
    [AddComponentMenu("Assetgen/Assetgen Character")]
    public sealed class AssetgenCharacter : MonoBehaviour
    {
        static readonly int SpeedId = Animator.StringToHash(CharacterDefinition.SpeedParameter);

        [SerializeField] CharacterDefinition definition;

        [Tooltip("Ground speed in metres per second. Drives the locomotion blend tree.")]
        [SerializeField, Min(0f)] float speed;

        [Tooltip("Move this transform forward at Speed. Handy for previews and simple NPCs; leave it off when a " +
                 "CharacterController, Rigidbody or NavMeshAgent moves the character.")]
        [SerializeField] bool moveForward;

        [Tooltip("Above the fastest clip's own speed, play the clips faster instead of letting the feet slide.")]
        [SerializeField] bool matchFootSpeed = true;

        [Tooltip("Cross-fade time, in seconds, for Play.")]
        [SerializeField, Min(0f)] float crossFade = 0.15f;

        [Tooltip("Each clip's sound (clips made from video), played from this object's AudioSource as the clip starts.")]
        [SerializeField] List<ClipSound> sounds = new List<ClipSound>();

        [Serializable]
        public struct ClipSound
        {
            public string clip;
            public AudioClip sound;
        }

        /// <summary>Any event of the playing clips: (clip, type, joint); type is "contact" or "impact".</summary>
        public event Action<string, string, string> ClipEvent;

        /// <summary>Part of the body came to rest on the ground, carried by this bone.</summary>
        public event Action<string> Footstep;

        /// <summary>A one-shot's action lands; the clip's <see cref="CharacterDefinition.Clip.phases"/> time its wind-up and recovery.</summary>
        public event Action<string> Impact;

        Animator m_Animator;
        AudioSource m_Audio;

        public CharacterDefinition Definition
        {
            get => definition;
            set => definition = value;
        }

        public Animator Animator => m_Animator != null ? m_Animator : m_Animator = GetComponent<Animator>();

        /// <summary>Ground speed in metres per second.</summary>
        public float Speed
        {
            get => speed;
            set => speed = Mathf.Max(0f, value);
        }

        public bool MoveForward
        {
            get => moveForward;
            set => moveForward = value;
        }

        public List<ClipSound> Sounds => sounds;

        /// <summary>The speed the walk clip was made at (0 without one).</summary>
        public float WalkSpeed => SpeedOf("walk");

        /// <summary>The speed the run clip was made at (0 without one).</summary>
        public float RunSpeed => SpeedOf("run");

        void Update()
        {
            var animator = Animator;
            if (animator.runtimeAnimatorController == null)
                return;
            animator.SetFloat(SpeedId, speed);
            if (matchFootSpeed && definition != null)
            {
                var top = definition.MaxLocomotionSpeed;
                animator.speed = top > 0f && speed > top ? Mathf.Min(speed / top, 2f) : 1f;
            }
            if (moveForward)
                transform.position += transform.forward * (speed * Time.deltaTime);
        }

        /// <summary>Cross-fade to a clip by name (see the definition's clips), or to "Locomotion".</summary>
        public bool Play(string clip, float fade = -1f)
        {
            var animator = Animator;
            if (!animator.HasState(0, Animator.StringToHash(clip)))
            {
                Debug.LogWarning($"{name} has no clip called '{clip}'", this);
                return false;
            }
            animator.CrossFadeInFixedTime(clip, fade < 0f ? crossFade : fade);
            return true;
        }

        /// <summary>Back to the locomotion blend, at <paramref name="metresPerSecond"/>.</summary>
        public void Locomote(float metresPerSecond)
        {
            Speed = metresPerSecond;
            Play(CharacterDefinition.LocomotionState);
        }

        /// <summary>"windup", "active" or "recovery" of a one-shot at <paramref name="time"/> seconds into it; "" without phases.</summary>
        public string PhaseOf(string clip, float time)
        {
            var info = definition != null ? definition.Find(clip) : null;
            return info != null ? info.phases.At(time) : "";
        }

        // Animation events the CharacterBuilder adds to clips made from video.
        void AssetgenContact(AnimationEvent e)
        {
            var (clip, joint) = Split(e.stringParameter);
            ClipEvent?.Invoke(clip, "contact", joint);
            Footstep?.Invoke(joint);
        }

        void AssetgenImpact(string clip)
        {
            ClipEvent?.Invoke(clip, "impact", "");
            Impact?.Invoke(clip);
        }

        void AssetgenSound(string clip)
        {
            foreach (var s in sounds)
            {
                if (s.clip != clip || s.sound == null)
                    continue;
                if (m_Audio == null && !TryGetComponent(out m_Audio))
                    return;
                m_Audio.PlayOneShot(s.sound);
                return;
            }
        }

        static (string clip, string joint) Split(string value)
        {
            var at = value?.IndexOf('|') ?? -1;
            return at < 0 ? (value ?? "", "") : (value.Substring(0, at), value.Substring(at + 1));
        }

        float SpeedOf(string role)
        {
            if (definition == null)
                return 0f;
            foreach (var c in definition.clips)
                if (c.role == role)
                    return c.speed;
            return 0f;
        }
    }
}
