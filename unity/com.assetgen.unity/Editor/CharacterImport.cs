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
    /// <summary>Brings characters into the project: from a server job, a zip, a folder or a .glb/.assetgen pair.</summary>
    public static class CharacterImport
    {
        /// <summary>Downloads a finished job's character, imports it and builds it. Returns the prefab path.</summary>
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
                                  ?? throw new InvalidDataException("the zip holds no .assetgen character file");
            var folder = NewFolder(Path.GetFileNameWithoutExtension(definitionEntry.Name));
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
        /// Imports a character from anywhere on disk: an export zip, a folder holding one, or a .glb or
        /// .assetgen file with its partner beside it. Returns the prefab path.
        /// </summary>
        public static string FromPath(string path)
        {
            path = Path.GetFullPath(path);
            if (path.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
                return FromZip(File.ReadAllBytes(path));
            string definition;
            if (Directory.Exists(path))
                definition = Directory.GetFiles(path, "*.assetgen", SearchOption.AllDirectories).FirstOrDefault()
                             ?? throw new FileNotFoundException($"no .assetgen character file in {path}");
            else
                definition = Path.ChangeExtension(path, ".assetgen");
            if (!File.Exists(definition))
                throw new FileNotFoundException($"no character file at {definition}");

            var projectPath = Path.GetFullPath(Directory.GetCurrentDirectory());
            if (definition.StartsWith(Path.Combine(projectPath, "Assets"), StringComparison.OrdinalIgnoreCase))
            {
                var assetPath = definition.Substring(projectPath.Length + 1).Replace('\\', '/');
                AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate);
                return Build(assetPath);
            }
            var stem = Path.GetFileNameWithoutExtension(definition);
            var folder = NewFolder(stem);
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
            var prefab = CharacterBuilder.Build(definitionPath);
            var path = AssetDatabase.GetAssetPath(prefab);
            Debug.Log($"Assetgen: imported {path}");
            return path;
        }

        /// <summary>A new, empty folder for a character under the characters folder: name, name 1, name 2, ...</summary>
        static string NewFolder(string name)
        {
            var parent = AssetgenSettings.instance.charactersFolder.TrimEnd('/');
            if (!parent.StartsWith("Assets", StringComparison.Ordinal))
                parent = "Assets/Characters";
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

        /// <summary>A character's prefab, by character name, character file path or prefab path.</summary>
        public static GameObject FindPrefab(string nameOrPath)
        {
            if (nameOrPath.EndsWith(".prefab", StringComparison.OrdinalIgnoreCase))
                return AssetDatabase.LoadAssetAtPath<GameObject>(nameOrPath);
            foreach (var (path, definition) in All())
            {
                if (path == nameOrPath || definition.name == nameOrPath || definition.characterName == nameOrPath)
                    return AssetDatabase.LoadAssetAtPath<GameObject>(CharacterBuilder.PrefabPath(path));
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
