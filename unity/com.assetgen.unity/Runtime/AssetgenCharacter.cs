using UnityEngine;

namespace Assetgen
{
    /// <summary>
    /// Plays an assetgen character. Set <see cref="Speed"/> (metres per second) and the locomotion
    /// blend picks idle, walk or run at the pace the clips were made for; call <see cref="Play"/>
    /// for anything else. One-shot clips return to locomotion when they end.
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

        Animator m_Animator;

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
