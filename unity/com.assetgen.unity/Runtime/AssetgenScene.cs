using System;
using System.Collections.Generic;
using UnityEngine;

namespace Assetgen
{
    /// <summary>
    /// An assetgen scene in the game: its <see cref="SceneDefinition"/>, its ground, and which object each
    /// placement's GameObject is, so game code can find every pine, or tell what a ray hit.
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("Assetgen/Assetgen Scene")]
    public sealed class AssetgenScene : MonoBehaviour
    {
        [Serializable]
        public struct Placement
        {
            public string objectId;
            public Transform node;
        }

        [SerializeField] SceneDefinition definition;
        [SerializeField] Transform ground;
        [SerializeField] List<Placement> placements = new List<Placement>();

        Dictionary<Transform, string> m_ObjectOf;

        public SceneDefinition Definition
        {
            get => definition;
            set => definition = value;
        }

        public Transform Ground
        {
            get => ground;
            set => ground = value;
        }

        public IReadOnlyList<Placement> Placements => placements;

        public void SetPlacements(IEnumerable<Placement> value)
        {
            placements.Clear();
            placements.AddRange(value);
            m_ObjectOf = null;
        }

        /// <summary>The placements of an object, by its id.</summary>
        public IEnumerable<Transform> InstancesOf(string objectId)
        {
            foreach (var p in placements)
                if (p.objectId == objectId && p.node != null)
                    yield return p.node;
        }

        /// <summary>
        /// The object <paramref name="target"/> belongs to: a placement, or anything under one (such as the
        /// collider a ray hit). Null for the ground and anything else.
        /// </summary>
        public SceneDefinition.SceneObject ObjectOf(Transform target)
        {
            if (definition == null)
                return null;
            if (m_ObjectOf == null)
            {
                m_ObjectOf = new Dictionary<Transform, string>();
                foreach (var p in placements)
                    if (p.node != null)
                        m_ObjectOf[p.node] = p.objectId;
            }
            for (var t = target; t != null && t != transform; t = t.parent)
                if (m_ObjectOf.TryGetValue(t, out var objectId))
                    return definition.Find(objectId);
            return null;
        }
    }
}
