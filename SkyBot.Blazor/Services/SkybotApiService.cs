using System.Net.Http.Json;
using Microsoft.AspNetCore.Components.Forms;
using SkyBot.Blazor.Models;

namespace SkyBot.Blazor.Services;

public class SkybotApiService
{
    private readonly HttpClient _http;

    public SkybotApiService(HttpClient http)
    {
        _http = http;
    }

    public async Task<ChatResponse> ChatAsync(string query, string? channel)
    {
        var request = new ChatRequest { Query = query, Channel = channel };
        var response = await _http.PostAsJsonAsync("/chat", request);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<ChatResponse>()
               ?? new ChatResponse { Answer = "No response received." };
    }

    public async Task<IngestResponse> IngestAsync(IBrowserFile file, string channel)
    {
        var content = new MultipartFormDataContent();
        var streamContent = new StreamContent(file.OpenReadStream(maxAllowedSize: 50 * 1024 * 1024));
        streamContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(file.ContentType);
        content.Add(streamContent, "file", file.Name);
        content.Add(new StringContent(channel), "channel");

        var response = await _http.PostAsync("/ingest", content);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<IngestResponse>()
               ?? new IngestResponse { Status = "unknown" };
    }

    public async Task<List<string>> GetChannelsAsync()
    {
        try
        {
            var result = await _http.GetFromJsonAsync<ChannelList>("/channels");
            return result?.Channels ?? new List<string>();
        }
        catch
        {
            return new List<string>();
        }
    }
}
