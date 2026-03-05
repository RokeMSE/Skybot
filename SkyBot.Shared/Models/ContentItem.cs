namespace SkyBot.Shared.Models;

/// <summary>
/// Represents a chunk of content extracted from a document.
/// Equivalent to Python's ContentItem dataclass.
/// </summary>
public class ContentItem
{
    public string Content { get; set; } = string.Empty;
    public string Type { get; set; } = "text"; // "text" or "image"
    public string Source { get; set; } = string.Empty;
    public int PageNum { get; set; }
    public string? ImagePath { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
}
