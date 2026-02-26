using System.Text.Json.Serialization;

namespace SkyBot.Blazor.Models;

public class ChannelList
{
    [JsonPropertyName("channels")]
    public List<string> Channels { get; set; } = new();
}
