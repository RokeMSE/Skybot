using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace SkyBot.Server.Services;

/// <summary>
/// Gemini LLM service using the Google GenAI REST API.
/// Port of Python's GeminiService (google-genai SDK).
/// </summary>
public class GeminiLlmService : ILlmService
{
    private readonly HttpClient _http;
    private readonly string _model;
    private readonly string _apiKey;

    public GeminiLlmService(string apiKey, string model, string? baseUrl = null)
    {
        _apiKey = apiKey;
        _model = model;
        _http = new HttpClient();

        if (!string.IsNullOrEmpty(baseUrl))
        {
            _http.BaseAddress = new Uri(baseUrl.TrimEnd('/'));
        }
        else
        {
            _http.BaseAddress = new Uri("https://generativelanguage.googleapis.com");
        }
    }

    public async Task<string> GenerateResponseAsync(string prompt, string? systemInstruction = null)
    {
        try
        {
            var requestBody = new Dictionary<string, object>();

            var contents = new List<object>
            {
                new { role = "user", parts = new[] { new { text = prompt } } }
            };
            requestBody["contents"] = contents;

            if (!string.IsNullOrEmpty(systemInstruction))
            {
                requestBody["systemInstruction"] = new
                {
                    parts = new[] { new { text = systemInstruction } }
                };
            }

            requestBody["generationConfig"] = new { temperature = 0.5 };

            var json = JsonSerializer.Serialize(requestBody);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var url = $"/v1beta/models/{_model}:generateContent?key={_apiKey}";
            var response = await _http.PostAsync(url, content);
            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(responseJson);

            var candidates = doc.RootElement.GetProperty("candidates");
            var text = candidates[0]
                .GetProperty("content")
                .GetProperty("parts")[0]
                .GetProperty("text")
                .GetString();

            return text ?? "No response generated.";
        }
        catch (Exception ex)
        {
            return $"Error generating response with Gemini: {ex.Message}";
        }
    }

    public async Task<string> AnalyzeImageAsync(Stream imageStream, string prompt)
    {
        try
        {
            using var ms = new MemoryStream();
            await imageStream.CopyToAsync(ms);
            var base64 = Convert.ToBase64String(ms.ToArray());

            var requestBody = new
            {
                contents = new[]
                {
                    new
                    {
                        role = "user",
                        parts = new object[]
                        {
                            new { text = prompt },
                            new
                            {
                                inline_data = new
                                {
                                    mime_type = "image/png",
                                    data = base64
                                }
                            }
                        }
                    }
                },
                generationConfig = new { temperature = 0.5 }
            };

            var json = JsonSerializer.Serialize(requestBody);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var url = $"/v1beta/models/{_model}:generateContent?key={_apiKey}";
            var response = await _http.PostAsync(url, content);
            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(responseJson);

            var text = doc.RootElement
                .GetProperty("candidates")[0]
                .GetProperty("content")
                .GetProperty("parts")[0]
                .GetProperty("text")
                .GetString();

            return text ?? "No analysis generated.";
        }
        catch (Exception ex)
        {
            return $"Error analyzing image with Gemini: {ex.Message}";
        }
    }
}
