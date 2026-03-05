using System.Text;
using System.Text.Json;
using System.Net.Http.Json;

namespace SkyBot.Server.Services;

/// <summary>
/// Ollama LLM service using the Ollama REST API directly.
/// Port of Python's OllamaService.
/// </summary>
public class OllamaLlmService : ILlmService
{
    private readonly HttpClient _http;
    private readonly string _model;

    public OllamaLlmService(string model, string endpoint = "http://localhost:11434")
    {
        _http = new HttpClient { BaseAddress = new Uri(endpoint.TrimEnd('/')) };
        _model = model;
    }

    public async Task<string> GenerateResponseAsync(string prompt, string? systemInstruction = null)
    {
        try
        {
            var messages = new List<object>();

            if (!string.IsNullOrEmpty(systemInstruction))
            {
                messages.Add(new { role = "system", content = systemInstruction });
            }

            messages.Add(new { role = "user", content = prompt });

            var body = new
            {
                model = _model,
                messages,
                stream = false
            };

            var resp = await _http.PostAsJsonAsync("/api/chat", body);
            resp.EnsureSuccessStatusCode();

            var json = await resp.Content.ReadFromJsonAsync<JsonElement>();
            return json.GetProperty("message").GetProperty("content").GetString() ?? "";
        }
        catch (Exception ex)
        {
            return $"Error generating response with Ollama: {ex.Message}";
        }
    }

    public async Task<string> AnalyzeImageAsync(Stream imageStream, string prompt)
    {
        try
        {
            using var ms = new MemoryStream();
            await imageStream.CopyToAsync(ms);
            var base64 = Convert.ToBase64String(ms.ToArray());

            var body = new
            {
                model = _model,
                messages = new[]
                {
                    new
                    {
                        role = "user",
                        content = prompt,
                        images = new[] { base64 }
                    }
                },
                stream = false
            };

            var resp = await _http.PostAsJsonAsync("/api/chat", body);
            resp.EnsureSuccessStatusCode();

            var json = await resp.Content.ReadFromJsonAsync<JsonElement>();
            return json.GetProperty("message").GetProperty("content").GetString() ?? "";
        }
        catch (Exception ex)
        {
            return $"Error analyzing image with Ollama: {ex.Message}";
        }
    }

    /// <summary>
    /// Generates embeddings using the Ollama embed endpoint.
    /// Used by VectorDbService for document embedding.
    /// </summary>
    public async Task<float[]> EmbedAsync(string text)
    {
        try
        {
            var body = new
            {
                model = "nomic-embed-text",
                input = text
            };

            var resp = await _http.PostAsJsonAsync("/api/embed", body);
            resp.EnsureSuccessStatusCode();

            var json = await resp.Content.ReadFromJsonAsync<JsonElement>();
            var embeddingsArr = json.GetProperty("embeddings");

            if (embeddingsArr.GetArrayLength() > 0)
            {
                var first = embeddingsArr[0];
                var result = new float[first.GetArrayLength()];
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = first[i].GetSingle();
                }
                return result;
            }

            return new float[768]; // fallback
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Embedding failed: {ex.Message}");
            return new float[768];
        }
    }
}
