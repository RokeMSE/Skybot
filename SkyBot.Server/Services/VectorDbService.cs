using System.Net.Http.Json;
using System.Text.Json;

namespace SkyBot.Server.Services;

/// <summary>
/// ChromaDB vector database service using the ChromaDB 1.x REST API.
/// API base path: /api/v1/databases/default_database/collections
/// 
/// Requires a running ChromaDB HTTP server:
///   chroma run --host 0.0.0.0 --port 8100 --path ./chroma_db
/// </summary>
public class VectorDbService
{
    private readonly HttpClient _http;
    private readonly string _collectionName;
    private readonly OllamaLlmService _ollamaEmbedder;
    private string? _collectionId;

    // ChromaDB 1.x (v2) API base path
    private const string ApiBase = "/api/v2/tenants/default_tenant/databases/default_database";

    public VectorDbService(string chromaEndpoint, string collectionName, OllamaLlmService ollamaEmbedder)
    {
        _http = new HttpClient { BaseAddress = new Uri(chromaEndpoint.TrimEnd('/')) };
        _collectionName = collectionName;
        _ollamaEmbedder = ollamaEmbedder;
    }

    private async Task<string> GetCollectionIdAsync()
    {
        if (_collectionId != null) return _collectionId;

        // List all collections and find by name
        var resp = await _http.GetAsync($"{ApiBase}/collections");
        if (resp.IsSuccessStatusCode)
        {
            var collections = await resp.Content.ReadFromJsonAsync<JsonElement>();
            foreach (var col in collections.EnumerateArray())
            {
                if (col.GetProperty("name").GetString() == _collectionName)
                {
                    _collectionId = col.GetProperty("id").GetString()!;
                    return _collectionId;
                }
            }
        }

        // Create collection if it doesn't exist
        var createBody = new { name = _collectionName, get_or_create = true };
        var createResp = await _http.PostAsJsonAsync($"{ApiBase}/collections", createBody);
        createResp.EnsureSuccessStatusCode();
        var createJson = await createResp.Content.ReadFromJsonAsync<JsonElement>();
        _collectionId = createJson.GetProperty("id").GetString()!;
        return _collectionId;
    }

    public async Task UpsertAsync(List<string> documents, List<Dictionary<string, object>> metadatas, List<string> ids)
    {
        var collectionId = await GetCollectionIdAsync();

        // Generate embeddings using Ollama nomic-embed-text
        var embeddings = new List<float[]>();
        foreach (var doc in documents)
        {
            var embedding = await _ollamaEmbedder.EmbedAsync(doc);
            embeddings.Add(embedding);
        }

        // Convert metadatas to Dictionary<string, string> for ChromaDB
        var metaList = metadatas.Select(m =>
            m.ToDictionary(kv => kv.Key, kv => kv.Value?.ToString() ?? "")
        ).ToList();

        var body = new
        {
            ids,
            documents,
            embeddings,
            metadatas = metaList
        };

        var resp = await _http.PostAsJsonAsync($"{ApiBase}/collections/{collectionId}/upsert", body);
        resp.EnsureSuccessStatusCode();
    }

    public async Task<(List<string> Documents, List<Dictionary<string, string>> Metadatas, List<string> Ids)>
        QueryAsync(string queryText, int nResults = 5, string? channelFilter = null)
    {
        var collectionId = await GetCollectionIdAsync();

        // Generate query embedding
        var queryEmbedding = await _ollamaEmbedder.EmbedAsync(queryText);

        var body = new Dictionary<string, object>
        {
            ["query_embeddings"] = new[] { queryEmbedding },
            ["n_results"] = nResults,
            ["include"] = new[] { "documents", "metadatas" }
        };

        if (!string.IsNullOrEmpty(channelFilter))
        {
            body["where"] = new Dictionary<string, object> { ["channel"] = channelFilter };
        }

        var resp = await _http.PostAsJsonAsync($"{ApiBase}/collections/{collectionId}/query", body);
        resp.EnsureSuccessStatusCode();

        var json = await resp.Content.ReadFromJsonAsync<JsonElement>();

        var documents = new List<string>();
        var metadatas = new List<Dictionary<string, string>>();
        var resultIds = new List<string>();

        if (json.TryGetProperty("ids", out var idsArr) && idsArr.GetArrayLength() > 0)
        {
            var innerIds = idsArr[0];
            var innerDocs = json.GetProperty("documents")[0];
            var innerMetas = json.GetProperty("metadatas")[0];

            for (int i = 0; i < innerIds.GetArrayLength(); i++)
            {
                resultIds.Add(innerIds[i].GetString() ?? "");
                documents.Add(innerDocs[i].GetString() ?? "");

                var meta = new Dictionary<string, string>();
                if (innerMetas[i].ValueKind == JsonValueKind.Object)
                {
                    foreach (var prop in innerMetas[i].EnumerateObject())
                    {
                        meta[prop.Name] = prop.Value.ToString();
                    }
                }
                metadatas.Add(meta);
            }
        }

        return (documents, metadatas, resultIds);
    }

    public async Task<(List<Dictionary<string, string>> Metadatas, List<string> Documents)>
        GetByFilterAsync(Dictionary<string, object> whereFilter)
    {
        var collectionId = await GetCollectionIdAsync();

        var body = new
        {
            where = whereFilter,
            include = new[] { "documents", "metadatas" }
        };

        var resp = await _http.PostAsJsonAsync($"{ApiBase}/collections/{collectionId}/get", body);
        resp.EnsureSuccessStatusCode();

        var json = await resp.Content.ReadFromJsonAsync<JsonElement>();

        var metadatas = new List<Dictionary<string, string>>();
        var documents = new List<string>();

        if (json.TryGetProperty("ids", out var idsArr))
        {
            var docs = json.GetProperty("documents");
            var metas = json.GetProperty("metadatas");

            for (int i = 0; i < idsArr.GetArrayLength(); i++)
            {
                documents.Add(docs[i].GetString() ?? "");

                var meta = new Dictionary<string, string>();
                if (metas[i].ValueKind == JsonValueKind.Object)
                {
                    foreach (var prop in metas[i].EnumerateObject())
                    {
                        meta[prop.Name] = prop.Value.ToString();
                    }
                }
                metadatas.Add(meta);
            }
        }

        return (metadatas, documents);
    }

    public async Task<List<string>> GetChannelsAsync()
    {
        try
        {
            var collectionId = await GetCollectionIdAsync();

            var body = new
            {
                include = new[] { "metadatas" }
            };

            var resp = await _http.PostAsJsonAsync($"{ApiBase}/collections/{collectionId}/get", body);
            resp.EnsureSuccessStatusCode();

            var json = await resp.Content.ReadFromJsonAsync<JsonElement>();
            var channels = new HashSet<string>();

            if (json.TryGetProperty("metadatas", out var metas))
            {
                for (int i = 0; i < metas.GetArrayLength(); i++)
                {
                    if (metas[i].ValueKind == JsonValueKind.Object &&
                        metas[i].TryGetProperty("channel", out var ch))
                    {
                        var chStr = ch.GetString();
                        if (!string.IsNullOrEmpty(chStr))
                            channels.Add(chStr);
                    }
                }
            }

            return channels.Order().ToList();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error fetching channels: {ex.Message}");
            return new List<string>();
        }
    }
}
