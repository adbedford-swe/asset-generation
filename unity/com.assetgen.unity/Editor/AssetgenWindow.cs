using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;

namespace Assetgen.Editor
{
    /// <summary>Window > Assetgen: describe a character (or give a picture), generate it, and it lands in the project.</summary>
    public sealed class AssetgenWindow : EditorWindow
    {
        [SerializeField] string prompt = "";
        [SerializeField] Texture2D image;
        [SerializeField] string imageFile = "";
        [SerializeField] string characterName = "";
        [SerializeField] float height = 1.8f;
        [SerializeField] List<string> motions = new List<string>();
        [SerializeField] Vector2 scroll;

        ReorderableList m_MotionList;
        ServerInfo m_Info;
        string m_InfoError;
        bool m_Busy;
        JobInfo[] m_Jobs = Array.Empty<JobInfo>();
        readonly Dictionary<string, Texture2D> m_Thumbnails = new Dictionary<string, Texture2D>();
        double m_NextRefresh;
        GUIStyle m_Wrapped, m_Muted, m_Title;

        static readonly (string label, string[] motions)[] Presets =
        {
            ("Humanoid: idle, walk, run, attack", AssetgenSettings.StarterMotions),
            ("Creature: idle, walk, run, roar", new[]
            {
                "An object stands idle and breathes.", "An object walks forward.", "An object runs forward.",
                "An object roars. [once]",
            }),
            ("NPC: idle, walk, wave, talk", new[]
            {
                "An object stands idle and breathes.", "An object walks forward.", "An object waves its hand. [once]",
                "An object talks with gestures.",
            }),
            ("Fighter: idle, walk, punch, kick, hit", new[]
            {
                "An object stands idle and breathes.", "An object walks forward.", "An object punches forward. [once]",
                "An object kicks forward. [once]", "An object gets hit and staggers back. [once]",
            }),
        };

        [MenuItem("Window/Assetgen", priority = 2100)]
        public static void Open() => GetWindow<AssetgenWindow>("Assetgen");

        void OnEnable()
        {
            titleContent = new GUIContent("Assetgen", AssetPreview.GetMiniTypeThumbnail(typeof(Avatar)));
            var settings = AssetgenSettings.instance;
            if (motions.Count == 0)
                motions = new List<string>(settings.defaultMotions.Count > 0 ? settings.defaultMotions : AssetgenSettings.StarterMotions.ToList());
            if (height <= 0f)
                height = settings.defaultHeight;
            m_MotionList = new ReorderableList(motions, typeof(string), true, true, true, true)
            {
                drawHeaderCallback = DrawMotionHeader,
                drawElementCallback = (rect, i, active, focused) =>
                    motions[i] = EditorGUI.TextField(new Rect(rect.x, rect.y + 2, rect.width, EditorGUIUtility.singleLineHeight), motions[i]),
                onAddCallback = list => motions.Add("An object "),
                elementHeight = EditorGUIUtility.singleLineHeight + 4,
            };
            JobTracker.Changed += OnJobChanged;
            Refresh();
        }

        void OnDisable() => JobTracker.Changed -= OnJobChanged;

        void OnJobChanged(JobInfo job)
        {
            for (var i = 0; i < m_Jobs.Length; i++)
                if (m_Jobs[i].id == job.id)
                    m_Jobs[i] = job;
            Repaint();
        }

        void Update()
        {
            var active = m_Jobs.Any(j => j.Active);
            if (EditorApplication.timeSinceStartup >= m_NextRefresh && (active || m_Info == null))
                Refresh();
        }

        async void Refresh()
        {
            m_NextRefresh = EditorApplication.timeSinceStartup + (m_Jobs.Any(j => j.Active) ? 2.0 : 10.0);
            var client = new AssetgenClient();
            try
            {
                m_Info = await client.InfoAsync();
                m_Jobs = await client.JobsAsync(12);
                m_InfoError = null;
                foreach (var job in m_Jobs)
                    LoadThumbnail(client, job);
            }
            catch (Exception e)
            {
                m_Info = null;
                m_InfoError = e is AssetgenException ? e.Message : $"no server at {client.BaseUrl}";
            }
            Repaint();
        }

        async void LoadThumbnail(AssetgenClient client, JobInfo job)
        {
            var url = job.urls?.preview ?? job.urls?.image;
            if (string.IsNullOrEmpty(url) || m_Thumbnails.ContainsKey(job.id + url))
                return;
            m_Thumbnails[job.id + url] = null;
            try
            {
                var texture = new Texture2D(2, 2) { hideFlags = HideFlags.HideAndDontSave };
                if (texture.LoadImage(await client.FileAsync(url)))
                    m_Thumbnails[job.id + url] = texture;
                Repaint();
            }
            catch (Exception)
            {
                // a missing thumbnail is not worth reporting
            }
        }

        void Styles()
        {
            if (m_Wrapped != null)
                return;
            m_Wrapped = new GUIStyle(EditorStyles.textArea) { wordWrap = true };
            m_Muted = new GUIStyle(EditorStyles.miniLabel) { wordWrap = true };
            m_Title = new GUIStyle(EditorStyles.boldLabel) { fontSize = 14 };
        }

        void OnGUI()
        {
            Styles();
            scroll = EditorGUILayout.BeginScrollView(scroll);
            DrawServer();
            EditorGUILayout.Space(8);
            DrawCreate();
            EditorGUILayout.Space(8);
            DrawJobs();
            EditorGUILayout.Space(8);
            DrawCharacters();
            EditorGUILayout.EndScrollView();
        }

        void DrawServer()
        {
            using (new EditorGUILayout.HorizontalScope(EditorStyles.toolbar))
            {
                var connected = m_Info != null;
                var dot = connected ? "<color=#4caf50>●</color>" : "<color=#9e9e9e>●</color>";
                var text = connected
                    ? $"{dot} assetgen {m_Info.version} · {string.Join(", ", m_Info.devices ?? Array.Empty<string>())} · images: " +
                      $"{m_Info.image_backend?.backend}{(m_Info.image_backend != null && m_Info.image_backend.ok ? "" : " (offline)")}"
                    : $"{dot} {m_InfoError ?? "connecting…"}";
                GUILayout.Label(text, new GUIStyle(EditorStyles.miniLabel) { richText = true });
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Web UI", EditorStyles.toolbarButton))
                    Application.OpenURL(AssetgenSettings.instance.serverUrl);
                if (GUILayout.Button("Refresh", EditorStyles.toolbarButton))
                    Refresh();
                if (GUILayout.Button(EditorGUIUtility.IconContent("_Popup"), EditorStyles.toolbarButton))
                    SettingsService.OpenProjectSettings("Project/Assetgen");
            }
            if (m_Info == null && m_InfoError != null)
                EditorGUILayout.HelpBox("Start the server with `python server.py` in the asset-generation folder, " +
                                        "or point Project Settings > Assetgen at a running one.", MessageType.Info);
        }

        void DrawCreate()
        {
            GUILayout.Label("New character", m_Title);
            EditorGUILayout.LabelField("Describe it", EditorStyles.miniBoldLabel);
            prompt = EditorGUILayout.TextArea(prompt, m_Wrapped, GUILayout.MinHeight(48));
            if (string.IsNullOrEmpty(prompt))
                GUILayout.Label("e.g. a goblin shaman with a lantern and a hooded cloak", m_Muted);

            using (new EditorGUILayout.HorizontalScope())
            {
                image = (Texture2D)EditorGUILayout.ObjectField(new GUIContent("Or from a picture"), image, typeof(Texture2D), false);
                if (GUILayout.Button("File…", GUILayout.Width(52)))
                {
                    var picked = EditorUtility.OpenFilePanel("Character picture", "", "png,jpg,jpeg,webp");
                    if (!string.IsNullOrEmpty(picked))
                    {
                        imageFile = picked;
                        image = null;
                    }
                }
            }
            if (!string.IsNullOrEmpty(imageFile) && image == null)
            {
                using (new EditorGUILayout.HorizontalScope())
                {
                    GUILayout.Label(Path.GetFileName(imageFile), m_Muted);
                    if (GUILayout.Button("×", EditorStyles.miniButton, GUILayout.Width(20)))
                        imageFile = "";
                }
            }

            using (new EditorGUILayout.HorizontalScope())
            {
                characterName = EditorGUILayout.TextField(new GUIContent("Name", "From the description when empty"), characterName);
                height = EditorGUILayout.FloatField(new GUIContent("Height (m)"), Mathf.Max(0.05f, height), GUILayout.MaxWidth(220));
            }
            EditorGUILayout.Space(4);
            m_MotionList.DoLayoutList();
            GUILayout.Label("Each motion is a clip. Start with \"An object\" and describe one thing; clips loop seamlessly " +
                            "unless the line ends in [once].", m_Muted);

            var picture = PictureBytes();
            using (new EditorGUI.DisabledScope(m_Info == null || m_Busy || (string.IsNullOrWhiteSpace(prompt) && picture == null)))
            {
                if (GUILayout.Button(m_Busy ? "Submitting…" : "Generate character", GUILayout.Height(30)))
                    Generate(picture);
            }
        }

        void DrawMotionHeader(Rect rect)
        {
            EditorGUI.LabelField(new Rect(rect.x, rect.y, rect.width - 70, rect.height), "Motions");
            if (GUI.Button(new Rect(rect.xMax - 66, rect.y + 1, 66, rect.height - 2), "Presets", EditorStyles.miniPullDown))
            {
                var menu = new GenericMenu();
                foreach (var (label, list) in Presets)
                    menu.AddItem(new GUIContent(label), false, () =>
                    {
                        motions.Clear();
                        motions.AddRange(list);
                        Repaint();
                    });
                menu.ShowAsContext();
            }
        }

        byte[] PictureBytes()
        {
            if (image != null)
            {
                var path = AssetDatabase.GetAssetPath(image);
                return string.IsNullOrEmpty(path) ? null : File.ReadAllBytes(path);
            }
            return !string.IsNullOrEmpty(imageFile) && File.Exists(imageFile) ? File.ReadAllBytes(imageFile) : null;
        }

        async void Generate(byte[] picture)
        {
            m_Busy = true;
            try
            {
                var request = new JobRequest
                {
                    prompt = prompt.Trim(),
                    name = characterName.Trim(),
                    height = height,
                    motions = motions.Where(m => !string.IsNullOrWhiteSpace(m)).Select(m => m.Trim()).ToArray(),
                };
                if (picture != null)
                    request.image = "data:image/png;base64," + Convert.ToBase64String(picture);
                var job = await new AssetgenClient().SubmitAsync(request);
                JobTracker.Track(job.id);
                m_Jobs = new[] { job }.Concat(m_Jobs.Where(j => j.id != job.id)).ToArray();
                m_NextRefresh = 0;
            }
            catch (Exception e)
            {
                EditorUtility.DisplayDialog("Assetgen", $"Could not start the job: {e.Message}", "OK");
            }
            finally
            {
                m_Busy = false;
                Repaint();
            }
        }

        void DrawJobs()
        {
            GUILayout.Label("On the server", m_Title);
            if (m_Jobs.Length == 0)
            {
                GUILayout.Label(m_Info == null ? "Not connected." : "No characters yet.", m_Muted);
                return;
            }
            foreach (var job in m_Jobs)
            {
                using (new EditorGUILayout.HorizontalScope(EditorStyles.helpBox))
                {
                    var url = job.urls?.preview ?? job.urls?.image;
                    m_Thumbnails.TryGetValue(job.id + (url ?? ""), out var thumbnail);
                    var box = GUILayoutUtility.GetRect(56, 56, GUILayout.Width(56), GUILayout.Height(56));
                    if (thumbnail != null)
                        GUI.DrawTexture(box, thumbnail, ScaleMode.ScaleAndCrop);
                    else
                        EditorGUI.DrawRect(box, new Color(0.18f, 0.18f, 0.2f));

                    using (new EditorGUILayout.VerticalScope())
                    {
                        GUILayout.Label(job.Label, EditorStyles.boldLabel);
                        var state = job.status == "running" ? $"running {job.stage}" : job.status;
                        if (!string.IsNullOrEmpty(job.error))
                            state += $": {job.error}";
                        GUILayout.Label($"{job.id} · {state}", m_Muted);
                        if (job.Active)
                        {
                            var bar = GUILayoutUtility.GetRect(18, 14, GUILayout.ExpandWidth(true));
                            EditorGUI.ProgressBar(bar, job.progress, $"{Mathf.RoundToInt(job.progress * 100)}%");
                        }
                    }
                    using (new EditorGUILayout.VerticalScope(GUILayout.Width(84)))
                    {
                        var imported = JobTracker.ImportedPrefab(job.id);
                        if (job.Active && GUILayout.Button("Cancel"))
                            Cancel(job.id);
                        if (job.Done && string.IsNullOrEmpty(imported) && GUILayout.Button("Import"))
                            Import(job.id);
                        if (!string.IsNullOrEmpty(imported) && GUILayout.Button("Select"))
                            Ping(imported);
                        if (!string.IsNullOrEmpty(imported) && GUILayout.Button("Add to scene"))
                            AddToScene(imported);
                    }
                }
            }
        }

        async void Cancel(string id)
        {
            try
            {
                await new AssetgenClient().CancelAsync(id);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Assetgen: {e.Message}");
            }
            m_NextRefresh = 0;
        }

        async void Import(string id)
        {
            try
            {
                Ping(await CharacterImport.FromJobAsync(id));
            }
            catch (Exception e)
            {
                EditorUtility.DisplayDialog("Assetgen", $"Import failed: {e.Message}", "OK");
            }
            Repaint();
        }

        void DrawCharacters()
        {
            GUILayout.Label("In this project", m_Title);
            var any = false;
            foreach (var (path, definition) in CharacterImport.All())
            {
                any = true;
                using (new EditorGUILayout.HorizontalScope())
                {
                    var loops = definition.clips.Count(c => c.loop);
                    GUILayout.Label(new GUIContent($"{definition.name}",
                            $"{definition.clips.Count} clips ({loops} looping) · {definition.height:0.00} m"),
                        GUILayout.MinWidth(80));
                    GUILayout.Label($"{definition.clips.Count} clips · {definition.height:0.00} m", m_Muted);
                    GUILayout.FlexibleSpace();
                    var prefab = CharacterBuilder.PrefabPath(path);
                    if (GUILayout.Button("Select", EditorStyles.miniButtonLeft, GUILayout.Width(56)))
                        Ping(File.Exists(prefab) ? prefab : path);
                    using (new EditorGUI.DisabledScope(!File.Exists(prefab)))
                    {
                        if (GUILayout.Button("Add to scene", EditorStyles.miniButtonRight, GUILayout.Width(90)))
                            AddToScene(prefab);
                    }
                }
            }
            if (!any)
                GUILayout.Label("None yet. Generated characters are imported into " +
                                $"{AssetgenSettings.instance.charactersFolder}.", m_Muted);
        }

        static void Ping(string assetPath)
        {
            var asset = AssetDatabase.LoadMainAssetAtPath(assetPath);
            if (asset == null)
                return;
            Selection.activeObject = asset;
            EditorGUIUtility.PingObject(asset);
        }

        public static GameObject AddToScene(string prefabPath, Vector3? position = null)
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
            if (prefab == null)
                return null;
            var instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
            Undo.RegisterCreatedObjectUndo(instance, $"Add {prefab.name}");
            if (position.HasValue)
            {
                instance.transform.position = position.Value;
            }
            else if (SceneView.lastActiveSceneView != null)
            {
                var pivot = SceneView.lastActiveSceneView.pivot;
                instance.transform.position = new Vector3(pivot.x, 0f, pivot.z);
            }
            Selection.activeGameObject = instance;
            return instance;
        }
    }
}
