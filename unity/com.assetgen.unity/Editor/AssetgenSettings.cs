using System.Collections.Generic;
using System.IO;
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

        [Tooltip("Where generated props and dioramas are imported, one folder each")]
        public string propsFolder = "Assets/Props";

        [Tooltip("Build the prefab (and a character's Animator Controller) whenever an asset's .glb or .assetgen file imports")]
        public bool buildOnImport = true;

        [Tooltip("Import assets generated from this project as soon as they finish")]
        public bool importWhenDone = true;

        [Min(0.01f)] public float defaultHeight = 1.8f;

        public List<string> defaultMotions = new List<string>(StarterMotions);

        [Tooltip("How each character is generated: sent with every job from this project")]
        public GenerationSettings generation = new GenerationSettings();

        /// <summary>The server's API token, if it needs one. Kept in this machine's preferences, never in the project.</summary>
        public static string Token
        {
            get => EditorPrefs.GetString("Assetgen.Token", "");
            set => EditorPrefs.SetString("Assetgen.Token", value ?? "");
        }

        public void SaveSettings() => Save(true);
    }

    /// <summary>The generation settings a job is made with; the defaults are the server's.</summary>
    [System.Serializable]
    public sealed class GenerationSettings
    {
        [Tooltip("The text-to-image model (sd-server: <recipe>:<file>, a recipe or a file). Empty: automatic")]
        public string imageModel = "";

        [Tooltip("Idol's image-to-splat sampling steps: fewer is faster and rougher")]
        [Min(1)] public int idolSteps = 50;

        [Tooltip("Motion sampler steps: 25-50 is close to UniMate's own solver at a fraction of the time")]
        [Min(1)] public int motionSteps = 32;

        [Tooltip("How strongly each motion follows its prompt (3 by default; 5-7 when a prompt is ignored)")]
        [Min(0f)] public float promptStrength = 3f;

        public int seed = 42;

        [Tooltip("Loop length in frames at 30 fps; 0 follows each motion's own cycle")]
        [Range(0, 600)] public int loopFrames;

        [Tooltip("Keep looping clips' travel instead of playing them in place")]
        public bool rootMotion;

        [Tooltip("Mesh face budget")]
        [Min(1000)] public int faceBudget = 40000;

        [Tooltip("Texture size in pixels")]
        public int textureSize = 2048;

        [Tooltip("Rigs sampled per character; the most plausible is kept")]
        [Min(1)] public int rigCandidates = 3;

        public JobSettings ToJob() => new JobSettings
        {
            idol_steps = idolSteps,
            flow_steps = motionSteps,
            cfg_scale = promptStrength,
            seed = seed,
            loop = loopFrames > 0 ? loopFrames.ToString() : "auto",
            root_motion = rootMotion,
            faces = faceBudget,
            texture = textureSize,
            rig_candidates = rigCandidates,
        };
    }

    static class AssetgenSettingsProvider
    {
        [SettingsProvider]
        static SettingsProvider Create() => new SettingsProvider("Project/Assetgen", SettingsScope.Project)
        {
            label = "Assetgen",
            keywords = new HashSet<string>(new[] { "assetgen", "character", "prop", "diorama", "server", "generate" }),
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
                EditorGUILayout.PropertyField(serialized.FindProperty("propsFolder"));
                EditorGUILayout.PropertyField(serialized.FindProperty("buildOnImport"));
                EditorGUILayout.PropertyField(serialized.FindProperty("importWhenDone"));
                EditorGUILayout.PropertyField(serialized.FindProperty("defaultHeight"));
                EditorGUILayout.PropertyField(serialized.FindProperty("defaultMotions"), true);
                EditorGUILayout.PropertyField(serialized.FindProperty("generation"), new GUIContent("Generation"), true);
                if (EditorGUI.EndChangeCheck())
                {
                    serialized.ApplyModifiedPropertiesWithoutUndo();
                    AssetgenSettings.Token = token;
                    settings.SaveSettings();
                }
                LocalServerGUI();
            },
        };

        static void LocalServerGUI()
        {
            EditorGUILayout.Space(12);
            EditorGUILayout.LabelField("Local server", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Unity can run the assetgen server itself. These settings belong to this machine.",
                EditorStyles.wordWrappedMiniLabel);

            using (new EditorGUILayout.HorizontalScope())
            {
                var detected = AssetgenServer.DetectedFolder;
                AssetgenServer.Folder = EditorGUILayout.TextField(
                    new GUIContent("assetgen folder", "The checkout with server.py. Empty: the one this package was installed from."),
                    AssetgenServer.Folder);
                if (GUILayout.Button("Browse…", GUILayout.Width(70)))
                {
                    var picked = EditorUtility.OpenFolderPanel("assetgen folder (with server.py)", AssetgenServer.ResolvedFolder ?? "", "");
                    if (!string.IsNullOrEmpty(picked))
                        AssetgenServer.Folder = picked;
                }
                if (string.IsNullOrEmpty(AssetgenServer.Folder) && detected != null)
                    GUILayout.Label("(this package's checkout)", EditorStyles.miniLabel, GUILayout.ExpandWidth(false));
            }
            AssetgenServer.Python = EditorGUILayout.TextField(
                new GUIContent("Python", "Empty: the checkout's .venv, then python on PATH."), AssetgenServer.Python);
            if (string.IsNullOrEmpty(AssetgenServer.Python))
                EditorGUILayout.LabelField(" ", $"uses {AssetgenServer.ResolvedPython(AssetgenServer.ResolvedFolder)}", EditorStyles.miniLabel);
            AssetgenServer.Arguments = EditorGUILayout.TextField(
                new GUIContent("Server arguments", "Added to server.py, e.g. --image-backend sdcpp --image-url http://127.0.0.1:1234"),
                AssetgenServer.Arguments);
            EditorGUILayout.LabelField(new GUIContent("Environment", "One KEY=VALUE per line, e.g. ASSETGEN_THIRD_PARTY=/data/third_party"));
            AssetgenServer.Environment = EditorGUILayout.TextArea(AssetgenServer.Environment, GUILayout.MinHeight(36));
            AssetgenServer.StartWhenNeeded = EditorGUILayout.Toggle(
                new GUIContent("Start when needed", "Start the server when you generate and none is answering"),
                AssetgenServer.StartWhenNeeded);
            AssetgenServer.StopWithEditor = EditorGUILayout.Toggle(
                new GUIContent("Stop with the Editor", "Stop the server Unity started when the Editor quits"),
                AssetgenServer.StopWithEditor);

            var reason = AssetgenServer.CannotStartReason;
            if (reason != null)
                EditorGUILayout.HelpBox(reason, MessageType.Info);
            using (new EditorGUILayout.HorizontalScope())
            {
                using (new EditorGUI.DisabledScope(reason != null || AssetgenServer.Owned))
                {
                    if (GUILayout.Button("Start"))
                        AssetgenWindow.StartServer();
                }
                using (new EditorGUI.DisabledScope(!AssetgenServer.Owned))
                {
                    if (GUILayout.Button("Stop"))
                        AssetgenServer.Stop();
                }
                using (new EditorGUI.DisabledScope(!File.Exists(AssetgenServer.LogPath)))
                {
                    if (GUILayout.Button("Open log"))
                        EditorUtility.OpenWithDefaultApp(AssetgenServer.LogPath);
                }
            }
            GUILayout.Label(AssetgenServer.Owned ? $"Running (started by this Editor, process {AssetgenServer.Process?.Id})" : "Not started by this Editor",
                EditorStyles.miniLabel);
        }
    }
}
