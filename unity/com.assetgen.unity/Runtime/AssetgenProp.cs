using UnityEngine;

namespace Assetgen
{
    /// <summary>
    /// An assetgen prop or diorama in the scene: its <see cref="PropDefinition"/>, with the size and
    /// footprint at this transform's scale, so game code and layout tools can space assets without
    /// them overlapping. The outline shows as a gizmo when selected.
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("Assetgen/Assetgen Prop")]
    public sealed class AssetgenProp : MonoBehaviour
    {
        [SerializeField] PropDefinition definition;

        public PropDefinition Definition
        {
            get => definition;
            set => definition = value;
        }

        public bool IsDiorama => definition != null && definition.IsDiorama;

        /// <summary>Width, height and depth in world units.</summary>
        public Vector3 Size
        {
            get
            {
                if (definition == null)
                    return Vector3.zero;
                var scale = transform.lossyScale;
                return Vector3.Scale(definition.size, new Vector3(Mathf.Abs(scale.x), Mathf.Abs(scale.y), Mathf.Abs(scale.z)));
            }
        }

        /// <summary>The farthest the asset reaches from its origin, seen from above, in world units.</summary>
        public float FootprintRadius => definition != null ? definition.footprintRadius * HorizontalScale : 0f;

        /// <summary>The same for its base: a tree's trunk rather than its canopy.</summary>
        public float BaseRadius => definition != null ? definition.baseRadius * HorizontalScale : 0f;

        float HorizontalScale
        {
            get
            {
                var scale = transform.lossyScale;
                return Mathf.Max(Mathf.Abs(scale.x), Mathf.Abs(scale.z));
            }
        }

        void OnDrawGizmosSelected()
        {
            if (definition == null || definition.outline.Count < 2)
                return;
            Gizmos.color = new Color(1f, 0.75f, 0.2f);
            Gizmos.matrix = transform.localToWorldMatrix;
            var outline = definition.outline;
            for (var i = 0; i < outline.Count; i++)
            {
                var a = outline[i];
                var b = outline[(i + 1) % outline.Count];
                Gizmos.DrawLine(new Vector3(a.x, 0f, a.y), new Vector3(b.x, 0f, b.y));
            }
        }
    }
}
