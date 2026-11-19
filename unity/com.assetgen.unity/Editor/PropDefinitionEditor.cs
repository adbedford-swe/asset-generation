using System.IO;
using UnityEditor;
using UnityEngine;

namespace Assetgen.Editor
{
    /// <summary>The inspector of a .assetgen prop or diorama: its size, collider and footprint, and the prefab built from it.</summary>
    [CustomEditor(typeof(PropDefinition))]
    sealed class PropDefinitionEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var definition = (PropDefinition)target;
            var path = AssetDatabase.GetAssetPath(definition);
            var prefabPath = PropBuilder.PrefabPath(path);
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);

            if (!string.IsNullOrEmpty(definition.prompt))
                EditorGUILayout.LabelField(definition.prompt, EditorStyles.wordWrappedLabel);
            var size = definition.size;
            EditorGUILayout.LabelField($"{(definition.IsDiorama ? "Diorama" : "Prop")} · {size.x:0.00} × {size.y:0.00} × " +
                                       $"{size.z:0.00} m · {definition.colliderType} collider", EditorStyles.miniLabel);
            EditorGUILayout.LabelField($"Footprint radius {definition.footprintRadius:0.00} m, base " +
                                       $"{definition.baseRadius:0.00} m · {definition.outline.Count}-point outline",
                EditorStyles.miniLabel);
            EditorGUILayout.Space(6);

            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button(prefab == null ? "Build prefab" : "Rebuild prefab"))
                {
                    try
                    {
                        EditorGUIUtility.PingObject(PropBuilder.Build(path));
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
            if (prefab == null && !File.Exists(PropBuilder.ModelPath(path, definition)))
                EditorGUILayout.HelpBox($"{definition.modelFile} is missing next to this file.", MessageType.Warning);
        }
    }
}
