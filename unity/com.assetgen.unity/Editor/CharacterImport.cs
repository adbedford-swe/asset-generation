using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace Assetgen.Editor
{
    /// <summary>
    /// Brings characters, props and dioramas into the project: from a server job, a zip, a folder or a
    /// .glb/.assetgen pair. Characters go under the characters folder, props and dioramas under the props folder.
    /// </summary>
    public static class CharacterImport
    {
        /// <summary>Downloads a finished job's asset, imports it and builds it. Returns the prefab path.</summary>
        public static async Task<string> FromJobAsync(string jobId, AssetgenClient client = null, CancellationToken ct = default)
        {
            client ??= new AssetgenClient();
            var zip = await client.ExportAsync(jobId, ct);
            var prefab = FromZip(zip);
            JobTracker.Imported(jobId, prefab);
            return prefab;
        }

        /// <summary>Imports an export zip (name/name.glb, name/name.assetgen, ...). Returns the prefab path.</summary>
        public static string FromZip(byte[] zip)
        {
            using var archive = new ZipArchive(new MemoryStream(zip), ZipArchiveMode.Read);
            var files = archive.Entries.Where(e => !string.IsNullOrEmpty(e.Name)).ToList();
            var definitionEntry = files.FirstOrDefault(e => e.Name.EndsWith(".assetgen", StringComparison.OrdinalIgnoreCase))
                                  ?? throw new InvalidDataException("the zip holds no .assetgen file");
            string json;
            using (var reader = new StreamReader(definitionEntry.Open()))
                json = reader.ReadToEnd();
            var folder = NewFolder(Path.GetFileNameWithoutExtension(definitionEntry.Name), IsProp(json));
            var root = Path.GetFullPath(folder);
            foreach (var entry in files)
            {
                var target = Path.GetFullPath(Path.Combine(folder, entry.Name));
                if (!target.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                    continue; // no paths out of the folder
                using var source = entry.Open();
                using var destination = File.Create(target);
                source.CopyTo(destination);
            }
            return ImportFolder(folder, definitionEntry.Name);
        }

        /// <summary>
        /// Imports a character, prop or diorama from anywhere on disk: an export zip, a folder holding one,
        /// or a .glb or .assetgen file with its partner beside it. Returns the prefab path.
        /// </summary>
        public static string FromPath(string path)
        {
            path = Path.GetFullPath(path);
            if (path.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
                return FromZip(File.ReadAllBytes(path));
            string definition;
            if (Directory.Exists(path))
                definition = Directory.GetFiles(path, "*.assetgen", SearchOption.AllDirectories).FirstOrDefault()
                             ?? throw new FileNotFoundException($"no .assetgen file in {path}");
            else
                definition = Path.ChangeExtension(path, ".assetgen");
            if (!File.Exists(definition))
                throw new FileNotFoundException($"no .assetgen file at {definition}");

            var projectPath = Path.GetFullPath(Directory.GetCurrentDirectory());
            if (definition.StartsWith(Path.Combine(projectPath, "Assets"), StringComparison.OrdinalIgnoreCase))
            {
                var assetPath = definition.Substring(projectPath.Length + 1).Replace('\\', '/');
                AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate);
                return Build(assetPath);
            }
            var stem = Path.GetFileNameWithoutExtension(definition);
            var folder = NewFolder(stem, IsProp(File.ReadAllText(definition)));
            var sourceFolder = Path.GetDirectoryName(definition) ?? ".";
            var kinds = new[] { ".glb", ".gltf", ".bin", ".assetgen", ".png" };
            foreach (var file in Directory.GetFiles(sourceFolder, stem + "*"))
                if (kinds.Contains(Path.GetExtension(file).ToLowerInvariant()))
                    File.Copy(file, Path.Combine(folder, Path.GetFileName(file)), true);
            return ImportFolder(folder, Path.GetFileName(definition));
        }

        static string ImportFolder(string folder, string definitionFile)
        {
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            return Build($"{folder}/{definitionFile}");
        }

        static string Build(string definitionPath)
        {
            var path = AssetDatabase.GetAssetPath(BuildPrefab(definitionPath));
            Debug.Log($"Assetgen: imported {path}");
            return path;
        }

        /// <summary>Builds (or rebuilds) the prefab of an imported .assetgen file, whatever it describes.</summary>
        public static GameObject BuildPrefab(string definitionPath) =>
            AssetDatabase.LoadAssetAtPath<PropDefinition>(definitionPath) != null
                ? PropBuilder.Build(definitionPath)
                : CharacterBuilder.Build(definitionPath);

        static bool IsProp(string json)
        {
            try
            {
                return CharacterImporter.FormatOf(json) == PropDefinition.Format;
            }
            catch (ArgumentException)
            {
                return false; // the importer reports what is wrong with the file
            }
        }

        /// <summary>A new, empty folder under the characters (or props) folder: name, name 1, name 2, ...</summary>
        static string NewFolder(string name, bool prop)
        {
            var settings = AssetgenSettings.instance;
            var parent = ((prop ? settings.propsFolder : settings.charactersFolder) ?? "").TrimEnd('/');
            if (!parent.StartsWith("Assets", StringComparison.Ordinal))
                parent = prop ? "Assets/Props" : "Assets/Characters";
            Directory.CreateDirectory(parent);
            var folder = $"{parent}/{name}";
            for (var k = 1; Directory.Exists(folder); k++)
                folder = $"{parent}/{name} {k}";
            Directory.CreateDirectory(folder);
            return folder;
        }

        /// <summary>Every character in the project: (character file path, definition).</summary>
        public static IEnumerable<(string path, CharacterDefinition definition)> All()
        {
            foreach (var guid in AssetDatabase.FindAssets("t:CharacterDefinition"))
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                var definition = AssetDatabase.LoadAssetAtPath<CharacterDefinition>(path);
                if (definition != null)
                    yield return (path, definition);
            }
        }

        /// <summary>Every prop and diorama in the project: (prop file path, definition).</summary>
        public static IEnumerable<(string path, PropDefinition definition)> AllProps()
        {
            foreach (var guid in AssetDatabase.FindAssets("t:PropDefinition"))
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                var definition = AssetDatabase.LoadAssetAtPath<PropDefinition>(path);
                if (definition != null)
                    yield return (path, definition);
            }
        }

        /// <summary>A character's or prop's prefab, by name, .assetgen file path or prefab path.</summary>
        public static GameObject FindPrefab(string nameOrPath)
        {
            if (nameOrPath.EndsWith(".prefab", StringComparison.OrdinalIgnoreCase))
                return AssetDatabase.LoadAssetAtPath<GameObject>(nameOrPath);
            foreach (var (path, definition) in All())
            {
                if (path == nameOrPath || definition.name == nameOrPath || definition.characterName == nameOrPath)
                    return AssetDatabase.LoadAssetAtPath<GameObject>(CharacterBuilder.PrefabPath(path));
            }
            foreach (var (path, definition) in AllProps())
            {
                if (path == nameOrPath || definition.name == nameOrPath || definition.propName == nameOrPath)
                    return AssetDatabase.LoadAssetAtPath<GameObject>(PropBuilder.PrefabPath(path));
            }
            return null;
        }
    }

    /// <summary>
    /// Follows the jobs this project started on the server and imports each one when it finishes
    /// (with "Import when done" on). Survives domain reloads and Editor restarts.
    /// </summary>
    [InitializeOnLoad]
    public static class JobTracker
    {
        const string PendingKey = "Assetgen.PendingJobs";
        const string ImportedKey = "Assetgen.ImportedJobs";
        static double s_NextPoll;
        static bool s_Polling;

        public static event Action<JobInfo> Changed;

        static JobTracker() => EditorApplication.update += Update;

        public static IReadOnlyList<string> Pending => Read(PendingKey).Keys.ToList();

        public static void Track(string jobId)
        {
            var pending = Read(PendingKey);
            pending[jobId] = "";
            Write(PendingKey, pending);
            s_NextPoll = 0;
        }

        public static string ImportedPrefab(string jobId) => Read(ImportedKey).TryGetValue(jobId, out var p) ? p : null;

        internal static void Imported(string jobId, string prefab)
        {
            var imported = Read(ImportedKey);
            imported[jobId] = prefab;
            Write(ImportedKey, imported);
            var pending = Read(PendingKey);
            if (pending.Remove(jobId))
                Write(PendingKey, pending);
        }

        static void Forget(string jobId)
        {
            var pending = Read(PendingKey);
            if (pending.Remove(jobId))
                Write(PendingKey, pending);
        }

        static async void Update()
        {
            if (s_Polling || EditorApplication.timeSinceStartup < s_NextPoll)
                return;
            s_NextPoll = EditorApplication.timeSinceStartup + 3.0;
            var pending = Pending;
            if (pending.Count == 0)
                return;
            s_Polling = true;
            try
            {
                var client = new AssetgenClient();
                foreach (var id in pending)
                {
                    JobInfo job;
                    try
                    {
                        job = await client.JobAsync(id);
                    }
                    catch (AssetgenException e) when (e.Status == 404)
                    {
                        Forget(id);
                        continue;
                    }
                    Changed?.Invoke(job);
                    if (job.Active)
                        continue;
                    if (job.Done && AssetgenSettings.instance.importWhenDone)
                        await CharacterImport.FromJobAsync(id, client);
                    else
                        Forget(id);
                }
            }
            catch (Exception e)
            {
                // The server may simply be down for a while; try again later without flooding the Console.
                s_NextPoll = EditorApplication.timeSinceStartup + 15.0;
                Debug.LogWarning($"Assetgen: {e.Message}");
            }
            finally
            {
                s_Polling = false;
            }
        }

        static Dictionary<string, string> Read(string key)
        {
            var map = new Dictionary<string, string>();
            foreach (var pair in (EditorUserSettings.GetConfigValue(key) ?? "").Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries))
            {
                var eq = pair.IndexOf('=');
                if (eq < 0) map[pair] = "";
                else map[pair.Substring(0, eq)] = pair.Substring(eq + 1);
            }
            return map;
        }

        static void Write(string key, Dictionary<string, string> map) =>
            EditorUserSettings.SetConfigValue(key, string.Join(";", map.Select(kv => $"{kv.Key}={kv.Value}")));
    }
}
