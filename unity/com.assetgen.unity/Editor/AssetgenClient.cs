using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Assetgen.Editor
{
    /// <summary>The assetgen server's HTTP API (see /docs on the server).</summary>
    public sealed class AssetgenClient
    {
        static readonly HttpClient s_Http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };

        public string BaseUrl { get; }
        public string Token { get; }

        public AssetgenClient(string baseUrl = null, string token = null)
        {
            BaseUrl = (baseUrl ?? AssetgenSettings.instance.serverUrl).TrimEnd('/');
            Token = token ?? AssetgenSettings.Token;
        }

        public Task<ServerInfo> InfoAsync(CancellationToken ct = default) => GetJsonAsync<ServerInfo>("info", ct, 5);

        /// <summary>The text-to-image models the server's image backend can load, and the one it would pick.</summary>
        public Task<ImageModels> ImageModelsAsync(CancellationToken ct = default) =>
            GetJsonAsync<ImageModels>("image-models", ct, 30);

        public Task<JobInfo> SubmitAsync(JobRequest request, CancellationToken ct = default) =>
            SendJsonAsync<JobInfo>(HttpMethod.Post, "jobs", JsonUtility.ToJson(request), ct);

        public Task<JobInfo> JobAsync(string id, CancellationToken ct = default) =>
            GetJsonAsync<JobInfo>($"jobs/{Uri.EscapeDataString(id)}", ct, 15);

        public async Task<JobInfo[]> JobsAsync(int limit = 20, CancellationToken ct = default) =>
            (await GetJsonAsync<JobList>($"jobs?limit={limit}", ct, 15)).jobs ?? Array.Empty<JobInfo>();

        public Task<JobInfo> CancelAsync(string id, CancellationToken ct = default) =>
            SendJsonAsync<JobInfo>(HttpMethod.Post, $"jobs/{Uri.EscapeDataString(id)}/cancel", "{}", ct);

        /// <summary>The job's asset packaged for Unity (a zip of name/name.glb, name.assetgen, ...).</summary>
        public Task<byte[]> ExportAsync(string id, CancellationToken ct = default) =>
            GetBytesAsync($"jobs/{Uri.EscapeDataString(id)}/export/unity", ct);

        /// <summary>Any file the server lists in a job's urls (a path starting with /api/v1/).</summary>
        public Task<byte[]> FileAsync(string apiPath, CancellationToken ct = default) =>
            GetBytesAsync(apiPath.StartsWith("/api/v1/", StringComparison.Ordinal) ? apiPath.Substring(8) : apiPath, ct);

        HttpRequestMessage Request(HttpMethod method, string path)
        {
            var request = new HttpRequestMessage(method, $"{BaseUrl}/api/v1/{path}");
            if (!string.IsNullOrEmpty(Token))
                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", Token);
            return request;
        }

        async Task<T> GetJsonAsync<T>(string path, CancellationToken ct, int timeoutSeconds)
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
            timeout.CancelAfter(TimeSpan.FromSeconds(timeoutSeconds));
            using var response = await s_Http.SendAsync(Request(HttpMethod.Get, path), timeout.Token);
            return await ReadAsync<T>(response);
        }

        async Task<T> SendJsonAsync<T>(HttpMethod method, string path, string json, CancellationToken ct)
        {
            var request = Request(method, path);
            request.Content = new StringContent(json, Encoding.UTF8, "application/json");
            using var response = await s_Http.SendAsync(request, ct);
            return await ReadAsync<T>(response);
        }

        async Task<byte[]> GetBytesAsync(string path, CancellationToken ct)
        {
            using var response = await s_Http.SendAsync(Request(HttpMethod.Get, path), ct);
            if (!response.IsSuccessStatusCode)
                throw new AssetgenException((int)response.StatusCode, await response.Content.ReadAsStringAsync());
            return await response.Content.ReadAsByteArrayAsync();
        }

        static async Task<T> ReadAsync<T>(HttpResponseMessage response)
        {
            var body = await response.Content.ReadAsStringAsync();
            if (!response.IsSuccessStatusCode)
                throw new AssetgenException((int)response.StatusCode, body);
            return JsonUtility.FromJson<T>(body);
        }
    }

    public sealed class AssetgenException : Exception
    {
        public int Status { get; }

        public AssetgenException(int status, string body) : base(Describe(status, body)) => Status = status;

        static string Describe(int status, string body)
        {
            var detail = JsonUtility.FromJson<ErrorBody>(string.IsNullOrEmpty(body) || body[0] != '{' ? "{}" : body)?.detail;
            return status == 401
                ? "the server wants an API token (Project Settings > Assetgen)"
                : $"server answered {status}: {(string.IsNullOrEmpty(detail) ? body : detail)}";
        }

#pragma warning disable 0649
        [Serializable] class ErrorBody { public string detail; }
#pragma warning restore 0649
    }

    /// <summary>What a job makes. Props and dioramas are static: no rig and no motions.</summary>
    public enum AssetKind
    {
        Character,
        Prop,
        Diorama,
    }

    public static class AssetKinds
    {
        /// <summary>The kind as the server and the .assetgen files name it: character, prop or diorama.</summary>
        public static string Id(this AssetKind kind) => kind.ToString().ToLowerInvariant();

        public static bool IsStatic(this AssetKind kind) => kind != AssetKind.Character;

        /// <summary>The height to ask for when none is given, in metres: the project's default for characters, the
        /// server's 1 m for props and dioramas.</summary>
        public static float DefaultHeight(this AssetKind kind) =>
            kind.IsStatic() ? 1f : AssetgenSettings.instance.defaultHeight;
    }

    // The API's JSON, for JsonUtility. Field names follow the server.
#pragma warning disable 0649
    [Serializable]
    public sealed class JobRequest
    {
        public string prompt;
        public string kind = "character";
        public string image;
        public string name;
        public string[] motions;
        public float height = 1.8f;
        public JobSettings settings;
        public ImageBackendRequest image_backend = new ImageBackendRequest();
    }

    /// <summary>Image backend overrides for one job; empty fields keep the server's.</summary>
    [Serializable]
    public sealed class ImageBackendRequest
    {
        public string model;
    }

    [Serializable]
    public sealed class ImageModels
    {
        public string backend;
        public string automatic;
        public string error;
        public ImageModelOption[] options;
        public LoadedImageModel loaded;
    }

    [Serializable] public sealed class ImageModelOption { public string id; public string label; public string recipe; }

    [Serializable] public sealed class LoadedImageModel { public string recipe; public string origin; }

    /// <summary>The server's job settings that the Editor sets (server.py: Settings).</summary>
    [Serializable]
    public sealed class JobSettings
    {
        public int idol_steps = 50;
        public int flow_steps = 32;
        public float cfg_scale = 3f;
        public int seed = 42;
        public string loop = "auto";
        public bool root_motion;
        public int faces = 40000;
        public int texture = 2048;
        public int rig_candidates = 3;
    }

    [Serializable]
    public sealed class JobInfo
    {
        public string id;
        public string name;
        public string kind;
        public string status;
        public string stage;
        public float progress;
        public string error;
        public JobInput input;
        public JobUrls urls;

        public bool Active => status == "queued" || status == "running";
        public bool Done => status == "done";
        public string Label => input != null && !string.IsNullOrEmpty(input.prompt) ? input.prompt : name;
    }

    [Serializable] public sealed class JobInput { public string prompt; public string image; public string ply; }

    [Serializable]
    public sealed class JobUrls
    {
        public string preview;
        public string image;
        public string model;
    }

    [Serializable] public sealed class JobList { public JobInfo[] jobs; }

    [Serializable]
    public sealed class ServerInfo
    {
        public string version;
        public string[] devices;
        public string[] default_motions;
        public ImageBackendInfo image_backend;
        public int busy;
        public int queued;
    }

    [Serializable] public sealed class ImageBackendInfo { public string backend; public string url; public bool ok; }
#pragma warning restore 0649
}
