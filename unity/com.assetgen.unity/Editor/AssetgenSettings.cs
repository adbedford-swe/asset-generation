using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace Assetgen.Editor
{
    /// <summary>Project settings (Project Settings > Assetgen), kept in ProjectSettings/AssetgenSettings.asset.</summary>
    [FilePath("ProjectSettings/AssetgenSettings.asset", FilePathAttribute.Location.ProjectFolder)]
    public sealed class AssetgenSettings : ScriptableSingleton<AssetgenSettings>
    {
        public static readonly string[] StarterMotions =
        {
            "An object stands idle and breathes.",
            "An object walks forward.",
            "An object runs forward.",
            "An object attacks forward. [once]",
        };

        [Tooltip("The assetgen server (python server.py)")]
        public string serverUrl = "http://127.0.0.1:7860";

        [Tooltip("Where generated characters are imported, one folder each")]
        public string charactersFolder = "Assets/Characters";

        [Tooltip("Build the prefab and Animator Controller whenever a character's .glb or .assetgen file imports")]
        public bool buildOnImport = true;

        [Tooltip("Import characters generated from this project as soon as they finish")]
        public bool importWhenDone = true;

        [Min(0.01f)] public float defaultHeight = 1.8f;

        public List<string> defaultMotions = new List<string>(StarterMotions);

        /// <summary>The server's API token, if it needs one. Kept in this machine's preferences, never in the project.</summary>
        public static string Token
        {
            get => EditorPrefs.GetString("Assetgen.Token", "");
            set => EditorPrefs.SetString("Assetgen.Token", value ?? "");
        }

        public void SaveSettings() => Save(true);
    }

    static class AssetgenSettingsProvider
    {
        [SettingsProvider]
        static SettingsProvider Create() => new SettingsProvider("Project/Assetgen", SettingsScope.Project)
        {
            label = "Assetgen",
            keywords = new HashSet<string>(new[] { "assetgen", "character", "server", "generate" }),
            guiHandler = _ =>
            {
                var settings = AssetgenSettings.instance;
                var serialized = new SerializedObject(settings);
                EditorGUI.BeginChangeCheck();
                EditorGUILayout.PropertyField(serialized.FindProperty("serverUrl"), new GUIContent("Server URL"));
                var token = EditorGUILayout.PasswordField(
                    new GUIContent("API token", "Only when the server was started with --token. Stored in EditorPrefs."),
                    AssetgenSettings.Token);
                EditorGUILayout.PropertyField(serialized.FindProperty("charactersFolder"));
                EditorGUILayout.PropertyField(serialized.FindProperty("buildOnImport"));
                EditorGUILayout.PropertyField(serialized.FindProperty("importWhenDone"));
                EditorGUILayout.PropertyField(serialized.FindProperty("defaultHeight"));
                EditorGUILayout.PropertyField(serialized.FindProperty("defaultMotions"), true);
                if (EditorGUI.EndChangeCheck())
                {
                    serialized.ApplyModifiedPropertiesWithoutUndo();
                    AssetgenSettings.Token = token;
                    settings.SaveSettings();
                }
            },
        };
    }
}
