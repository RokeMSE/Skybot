using System.Text.Json.Serialization;

namespace SkyBot.Blazor.Models;

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
