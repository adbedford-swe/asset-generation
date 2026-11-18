using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace Assetgen.Editor
{
    /// <summary>
    /// Starts, watches and stops a local assetgen server (<c>server.py</c>) for this Editor, so
    /// generating from the window or the Unity CLI needs nothing started by hand. Where to find
    /// assetgen and which Python runs it are this machine's settings (EditorPrefs), not the project's.
    /// </summary>
    [InitializeOnLoad]
    public static class AssetgenServer
    {
        const string PidKey = "Assetgen.ServerPid";

        public static event Action Changed;

        static AssetgenServer()
        {
            EditorApplication.quitting += () =>
            {
                if (StopWithEditor)
                    Stop();
            };
        }

        /// <summary>The assetgen checkout (the folder with server.py); empty to use the one this package came from.</summary>
        public static string Folder
        {
            get => EditorPrefs.GetString("Assetgen.ServerFolder", "");
            set => EditorPrefs.SetString("Assetgen.ServerFolder", value ?? "");
        }

        /// <summary>The Python that runs the server; empty to use the checkout's .venv, then python on PATH.</summary>
        public static string Python
        {
            get => EditorPrefs.GetString("Assetgen.ServerPython", "");
            set => EditorPrefs.SetString("Assetgen.ServerPython", value ?? "");
        }

        /// <summary>Extra environment for the server, one KEY=VALUE per line.</summary>
        public static string Environment
        {
            get => EditorPrefs.GetString("Assetgen.ServerEnvironment", "");
            set => EditorPrefs.SetString("Assetgen.ServerEnvironment", value ?? "");
        }

        /// <summary>Extra server.py arguments, e.g. --image-backend sdcpp --image-url http://127.0.0.1:1234.</summary>
        public static string Arguments
        {
            get => EditorPrefs.GetString("Assetgen.ServerArguments", "");
            set => EditorPrefs.SetString("Assetgen.ServerArguments", value ?? "");
        }

        /// <summary>Start the server when something needs it and nothing answers.</summary>
        public static bool StartWhenNeeded
        {
            get => EditorPrefs.GetBool("Assetgen.StartWhenNeeded", true);
            set => EditorPrefs.SetBool("Assetgen.StartWhenNeeded", value);
        }

        /// <summary>Stop the server this Editor started when the Editor quits.</summary>
        public static bool StopWithEditor
        {
            get => EditorPrefs.GetBool("Assetgen.StopWithEditor", true);
            set => EditorPrefs.SetBool("Assetgen.StopWithEditor", value);
        }

        public static string LogPath => Path.GetFullPath("Library/Assetgen/server.log");

        /// <summary>The checkout this package was installed from, when it was installed from one (a file: reference).</summary>
        public static string DetectedFolder
        {
            get
            {
                var package = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(AssetgenServer).Assembly);
                if (package == null || string.IsNullOrEmpty(package.resolvedPath))
                    return null;
                var root = Path.GetFullPath(Path.Combine(package.resolvedPath, "..", ".."));
                return File.Exists(Path.Combine(root, "server.py")) ? root : null;
            }
        }

        public static string ResolvedFolder => !string.IsNullOrEmpty(Folder) ? Folder : DetectedFolder;

        public static string ResolvedPython(string folder)
        {
            if (!string.IsNullOrEmpty(Python))
                return Python;
            foreach (var venv in new[] { ".venv/Scripts/python.exe", ".venv/bin/python" })
            {
                var candidate = Path.Combine(folder ?? "", venv);
                if (File.Exists(candidate))
                    return candidate;
            }
            return Application.platform == RuntimePlatform.WindowsEditor ? "python" : "python3";
        }

        /// <summary>The server process this Editor started, while it runs.</summary>
        public static Process Process
        {
            get
            {
                var pid = SessionState.GetInt(PidKey, 0);
                if (pid == 0)
                    return null;
                try
                {
                    var process = System.Diagnostics.Process.GetProcessById(pid);
                    if (!process.HasExited)
                        return process;
                }
                catch (ArgumentException)
                {
                    // gone
                }
                SessionState.EraseInt(PidKey);
                return null;
            }
        }

        /// <summary>Whether this Editor started the server and it is still running.</summary>
        public static bool Owned => Process != null;

        /// <summary>Why the server can't be started from here, or null when it can.</summary>
        public static string CannotStartReason
        {
            get
            {
                var folder = ResolvedFolder;
                if (string.IsNullOrEmpty(folder) || !File.Exists(Path.Combine(folder, "server.py")))
                    return "Set the assetgen folder (the one with server.py) in Project Settings > Assetgen.";
                if (!Uri.TryCreate(AssetgenSettings.instance.serverUrl, UriKind.Absolute, out var uri) || !uri.IsLoopback)
                    return $"{AssetgenSettings.instance.serverUrl} is not on this machine, so Unity can't start it.";
                return null;
            }
        }

        public static void Start()
        {
            if (Owned)
                return;
            var reason = CannotStartReason;
            if (reason != null)
                throw new InvalidOperationException(reason);
            var folder = ResolvedFolder;
            var port = new Uri(AssetgenSettings.instance.serverUrl).Port;
            Directory.CreateDirectory(Path.GetDirectoryName(LogPath) ?? "Library/Assetgen");
            File.AppendAllText(LogPath, $"\n== {DateTime.Now:yyyy-MM-dd HH:mm:ss} starting the assetgen server from {folder}\n");
            var start = new ProcessStartInfo(ResolvedPython(folder))
            {
                WorkingDirectory = folder,
                Arguments = $"server.py --host 127.0.0.1 --port {port} --log-file \"{LogPath}\" {Arguments}".Trim(),
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            foreach (var (key, value) in EnvironmentVariables())
                start.EnvironmentVariables[key] = value;
            start.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
            System.Diagnostics.Process process;
            try
            {
                process = System.Diagnostics.Process.Start(start);
            }
            catch (System.ComponentModel.Win32Exception e)
            {
                throw new InvalidOperationException(
                    $"could not run {start.FileName} ({e.Message}); set the Python in Project Settings > Assetgen");
            }
            SessionState.SetInt(PidKey, process.Id);
            Changed?.Invoke();
        }

        /// <summary>Stops the server this Editor started (and the stage it is running). A server started elsewhere is left alone.</summary>
        public static void Stop()
        {
            var process = Process;
            if (process == null)
                return;
            try
            {
                if (Application.platform == RuntimePlatform.WindowsEditor)
                {
                    // The venv's python.exe is a launcher with the server as its child, and stages are
                    // children of the server: take the whole tree.
                    using var kill = System.Diagnostics.Process.Start(new ProcessStartInfo("taskkill", $"/T /F /PID {process.Id}")
                    {
                        UseShellExecute = false,
                        CreateNoWindow = true,
                    });
                    kill?.WaitForExit(15000);
                }
                else
                {
                    // SIGTERM: the server cancels its jobs, stopping their stages, then exits.
                    using var term = System.Diagnostics.Process.Start("kill", $"-TERM {process.Id}");
                    term?.WaitForExit(5000);
                    if (!process.WaitForExit(30000))
                        process.Kill();
                }
            }
            catch (Exception e) when (e is InvalidOperationException || e is System.ComponentModel.Win32Exception)
            {
                // already gone
            }
            finally
            {
                SessionState.EraseInt(PidKey);
                Changed?.Invoke();
            }
        }

        public static async Task RestartAsync(CancellationToken ct = default)
        {
            Stop();
            Start();
            await WaitUntilReadyAsync(new AssetgenClient(), ct);
        }

        /// <summary>
        /// Makes sure a server answers at the configured URL: if none does and starting one is allowed,
        /// starts it and waits until it is ready (the first start imports PyTorch and Gradio: tens of seconds).
        /// </summary>
        public static async Task EnsureRunningAsync(CancellationToken ct = default)
        {
            var client = new AssetgenClient();
            if (await AnswersAsync(client, ct))
                return;
            if (!Owned)
            {
                if (!StartWhenNeeded)
                    throw new InvalidOperationException($"no assetgen server at {client.BaseUrl}");
                Start();
            }
            await WaitUntilReadyAsync(client, ct);
        }

        public static async Task WaitUntilReadyAsync(AssetgenClient client, CancellationToken ct = default, double seconds = 300)
        {
            var end = EditorApplication.timeSinceStartup + seconds;
            var next = 0.0;
            while (EditorApplication.timeSinceStartup < end)
            {
                ct.ThrowIfCancellationRequested();
                if (!Owned)
                    throw new InvalidOperationException($"the assetgen server stopped while starting:\n{LogTail(12)}");
                if (EditorApplication.timeSinceStartup >= next)
                {
                    if (await AnswersAsync(client, ct))
                    {
                        Changed?.Invoke();
                        return;
                    }
                    next = EditorApplication.timeSinceStartup + 1.0;
                }
                await Task.Yield();
            }
            throw new TimeoutException($"the assetgen server did not answer within {seconds:0} s; see {LogPath}");
        }

        static async Task<bool> AnswersAsync(AssetgenClient client, CancellationToken ct)
        {
            try
            {
                await client.InfoAsync(ct);
                return true;
            }
            catch (Exception e) when (!(e is OperationCanceledException) || !ct.IsCancellationRequested)
            {
                return false;
            }
        }

        /// <summary>The last lines of the server's log.</summary>
        public static string LogTail(int lines)
        {
            if (!File.Exists(LogPath))
                return "";
            using var stream = new FileStream(LogPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            using var reader = new StreamReader(stream);
            var all = reader.ReadToEnd().Split('\n');
            return string.Join("\n", all.Skip(Math.Max(0, all.Length - lines)));
        }

        static IEnumerable<(string, string)> EnvironmentVariables()
        {
            foreach (var line in Environment.Split('\n'))
            {
                var trimmed = line.Trim();
                var eq = trimmed.IndexOf('=');
                if (eq > 0 && !trimmed.StartsWith("#", StringComparison.Ordinal))
                    yield return (trimmed.Substring(0, eq).Trim(), trimmed.Substring(eq + 1).Trim());
            }
        }
    }
}
