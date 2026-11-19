using System.IO;
using UnityEditor;
using UnityEngine;

namespace Assetgen.Editor
{
    /// <summary>The inspector of a .assetgen scene: its objects and placements, and the prefab built from it.</summary>
    [CustomEditor(typeof(SceneDefinition))]
    sealed class SceneDefinitionEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var definition = (SceneDefinition)target;
            var path = AssetDatabase.GetAssetPath(definition);
            var prefabPath = SceneBuilder.PrefabPath(path);
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);

            EditorGUILayout.LabelField($"Ground {definition.size.x:0.#} × {definition.size.y:0.#} m · " +
                                       $"{definition.objects.Count} objects · {definition.instances.Count} placements",
                EditorStyles.miniLabel);
            EditorGUILayout.Space(6);

            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.Label("Object", EditorStyles.miniBoldLabel, GUILayout.MinWidth(120));
                GUILayout.Label("Kind", EditorStyles.miniBoldLabel, GUILayout.Width(56));
                GUILayout.Label("Height", EditorStyles.miniBoldLabel, GUILayout.Width(56));
                GUILayout.Label("Collider", EditorStyles.miniBoldLabel, GUILayout.Width(56));
                GUILayout.Label("Placed", EditorStyles.miniBoldLabel, GUILayout.Width(44));
            }
            foreach (var o in definition.objects)
            {
                using (new EditorGUILayout.HorizontalScope())
                {
                    GUILayout.Label(new GUIContent(o.id, o.prompt), GUILayout.MinWidth(120));
                    GUILayout.Label(o.kind, GUILayout.Width(56));
                    GUILayout.Label($"{o.height:0.00} m", GUILayout.Width(56));
                    GUILayout.Label(o.colliderType.ToString(), GUILayout.Width(56));
                    GUILayout.Label(o.count.ToString(), GUILayout.Width(44));
                }
            }
            foreach (var warning in definition.warnings)
                EditorGUILayout.HelpBox(warning, MessageType.Warning);
            EditorGUILayout.Space(6);

            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button(prefab == null ? "Build prefab" : "Rebuild prefab"))
                {
                    try
                    {
                        EditorGUIUtility.PingObject(SceneBuilder.Build(path));
                    }
                    catch (System.Exception e)
                    {
                        Debug.LogError($"Assetgen: {e.Message}");
                    }
                }
                using (new EditorGUI.DisabledScope(prefab == null))
                {
                    if (GUILayout.Button("Select prefab"))
                    {
                        Selection.activeObject = prefab;
                        EditorGUIUtility.PingObject(prefab);
                    }
                    if (GUILayout.Button("Add to scene"))
                        AssetgenWindow.AddToScene(prefabPath);
                }
            }
            if (prefab == null && !File.Exists(SceneBuilder.ModelPath(path, definition)))
                EditorGUILayout.HelpBox($"{definition.modelFile} is missing next to this file.", MessageType.Warning);
        }
    }
}
