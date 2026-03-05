using System.Text.Json.Serialization;

namespace SkyBot.Shared.Models;

public class ChatRequest
{
    [JsonPropertyName("query")]
    public string Query { get; set; } = string.Empty;

    [JsonPropertyName("channel")]
    public string? Channel { get; set; }
}

public class ChatResponse
{
    [JsonPropertyName("answer")]
    public string Answer { get; set; } = string.Empty;

    [JsonPropertyName("citations")]
    public List<Citation> Citations { get; set; } = new();

    [JsonPropertyName("images")]
    public List<string> Images { get; set; } = new();
}

public class Citation
{
    [JsonPropertyName("source")]
    public string Source { get; set; } = "Unknown";

    [JsonPropertyName("page")]
    public string Page { get; set; } = "?";

    [JsonPropertyName("type")]
    public string? Type { get; set; }
}

public class IngestResponse
{
    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("file")]
    public string File { get; set; } = string.Empty;

    [JsonPropertyName("chunks")]
    public int Chunks { get; set; }

    [JsonPropertyName("ingest_id")]
    public string IngestId { get; set; } = string.Empty;
}

public class ChannelList
{
    [JsonPropertyName("channels")]
    public List<string> Channels { get; set; } = new();
}
