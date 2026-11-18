using System.IO;
using UnityEditor;
using UnityEngine;

namespace Assetgen.Editor
{
    /// <summary>The inspector of a .assetgen character: its clips, and the prefab built from it.</summary>
    [CustomEditor(typeof(CharacterDefinition))]
    sealed class CharacterDefinitionEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var definition = (CharacterDefinition)target;
            var path = AssetDatabase.GetAssetPath(definition);
            var prefabPath = CharacterBuilder.PrefabPath(path);
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);

            if (!string.IsNullOrEmpty(definition.prompt))
                EditorGUILayout.LabelField(definition.prompt, EditorStyles.wordWrappedLabel);
            EditorGUILayout.LabelField($"{definition.height:0.00} m tall · {definition.jointCount} joints · " +
                                       $"{definition.clips.Count} clips", EditorStyles.miniLabel);
            EditorGUILayout.Space(6);

            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.Label("Clip", EditorStyles.miniBoldLabel, GUILayout.MinWidth(120));
                GUILayout.Label("Role", EditorStyles.miniBoldLabel, GUILayout.Width(56));
                GUILayout.Label("Loop", EditorStyles.miniBoldLabel, GUILayout.Width(40));
                GUILayout.Label("Length", EditorStyles.miniBoldLabel, GUILayout.Width(50));
                GUILayout.Label("Speed", EditorStyles.miniBoldLabel, GUILayout.Width(64));
            }
            foreach (var clip in definition.clips)
            {
                using (new EditorGUILayout.HorizontalScope())
                {
                    GUILayout.Label(new GUIContent(clip.name, clip.prompt), GUILayout.MinWidth(120));
                    GUILayout.Label(clip.role, GUILayout.Width(56));
                    GUILayout.Label(clip.loop ? "yes" : "once", GUILayout.Width(40));
                    GUILayout.Label($"{clip.duration:0.00} s", GUILayout.Width(50));
                    GUILayout.Label(clip.speed > 0f ? $"{clip.speed:0.00} m/s" : "", GUILayout.Width(64));
                }
            }
            if (definition.locomotion.Count > 1)
            {
                EditorGUILayout.Space(4);
                EditorGUILayout.HelpBox(
                    "The \"Locomotion\" state blends " + string.Join(" → ", definition.locomotion.ConvertAll(p => $"{p.clip} ({p.speed:0.00})")) +
                    " by the Animator's Speed parameter (m/s). Set AssetgenCharacter.Speed, or the parameter, to the " +
                    "character's ground speed.", MessageType.None);
            }
            EditorGUILayout.Space(6);

            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button(prefab == null ? "Build prefab" : "Rebuild prefab"))
                {
                    try
                    {
                        EditorGUIUtility.PingObject(CharacterBuilder.Build(path));
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
            if (!string.IsNullOrEmpty(definition.license))
                EditorGUILayout.HelpBox(definition.license, MessageType.Info);
            if (prefab == null && !File.Exists(CharacterBuilder.ModelPath(path, definition)))
                EditorGUILayout.HelpBox($"{definition.modelFile} is missing next to this file.", MessageType.Warning);
        }
    }
}
