using SkyBot.Shared.Models;

namespace SkyBot.Server.Extractors;

/// <summary>
/// Extracts content from plain text files (.txt, .md, .log).
/// Port of Python's TextExtractor.
/// </summary>
public class TextExtractor : IExtractor
{
    public List<ContentItem> Extract(string filePath)
    {
        var items = new List<ContentItem>();
        var filename = Path.GetFileName(filePath);
        var text = File.ReadAllText(filePath);

        if (!string.IsNullOrWhiteSpace(text))
        {
            items.Add(new ContentItem
            {
                Content = text,
                Type = "text",
                Source = filename,
                PageNum = 1
            });
        }

        return items;
    }
}
