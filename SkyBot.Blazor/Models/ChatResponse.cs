using System.Text.Json.Serialization;

namespace SkyBot.Blazor.Models;

public class ChatResponse
{
    [JsonPropertyName("answer")]
    public string Answer { get; set; } = string.Empty;

    [JsonPropertyName("citations")]
    public List<Citation>? Citations { get; set; }

    [JsonPropertyName("images")]
    public List<string>? Images { get; set; }
}

public class Citation
{
    [JsonPropertyName("source")]
    public string Source { get; set; } = "Unknown";

    [JsonPropertyName("page")]
    public string Page { get; set; } = "?";
}
