namespace SkyBot.Blazor.Models;

public class ChatRequest
{
    public string Query { get; set; } = string.Empty;
    public string? Channel { get; set; }
}
