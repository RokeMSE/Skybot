using SkyBot.Shared.Models;
using HtmlAgilityPack;
using SixLabors.ImageSharp;

namespace SkyBot.Server.Extractors;

/// <summary>
/// Extracts text and embedded base64 images from HTML files.
/// Port of Python's HTMLExtractor (BeautifulSoup).
/// </summary>
public class HtmlExtractor : IExtractor
{
    private readonly string _imageStoreDir;

    public HtmlExtractor(string imageStoreDir)
    {
        _imageStoreDir = imageStoreDir;
        Directory.CreateDirectory(_imageStoreDir);
    }

    public List<ContentItem> Extract(string filePath)
    {
        var items = new List<ContentItem>();
        var filename = Path.GetFileName(filePath);
        var htmlContent = File.ReadAllText(filePath);

        var doc = new HtmlDocument();
        doc.LoadHtml(htmlContent);

        // Remove script, style, nav, footer, header
        var tagsToRemove = new[] { "script", "style", "nav", "footer", "header" };
        foreach (var tag in tagsToRemove)
        {
            var nodes = doc.DocumentNode.SelectNodes($"//{tag}");
            if (nodes != null)
            {
                foreach (var node in nodes)
                    node.Remove();
            }
        }

        // --- 1. Text Extraction ---
        var text = doc.DocumentNode.InnerText
            .Replace("\r\n", "\n")
            .Replace("\r", "\n");
        // Collapse multiple newlines
        while (text.Contains("\n\n\n"))
            text = text.Replace("\n\n\n", "\n\n");
        text = text.Trim();

        if (!string.IsNullOrEmpty(text))
        {
            items.Add(new ContentItem
            {
                Content = text,
                Type = "text",
                Source = filename,
                PageNum = 1
            });
        }

        // --- 2. Embedded Base64 Image Extraction ---
        var imgNodes = doc.DocumentNode.SelectNodes("//img[@src]");
        if (imgNodes != null)
        {
            foreach (var imgNode in imgNodes)
            {
                var src = imgNode.GetAttributeValue("src", "");
                if (!src.StartsWith("data:image")) continue;

                try
                {
                    var commaIndex = src.IndexOf(',');
                    if (commaIndex < 0) continue;

                    var base64Data = src.Substring(commaIndex + 1);
                    var imageBytes = Convert.FromBase64String(base64Data);

                    using var memStream = new MemoryStream(imageBytes);
                    using var img = SixLabors.ImageSharp.Image.Load(memStream);

                    if (img.Width < 100 || img.Height < 100)
                        continue;

                    var imageId = $"{Guid.NewGuid()}.png";
                    var savePath = Path.Combine(_imageStoreDir, imageId);
                    img.SaveAsPng(savePath);

                    items.Add(new ContentItem
                    {
                        Content = "[[Image extracted from HTML]]",
                        Type = "image",
                        Source = filename,
                        PageNum = 1,
                        ImagePath = savePath,
                        Metadata = new Dictionary<string, object>
                        {
                            ["width"] = img.Width,
                            ["height"] = img.Height
                        }
                    });
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to extract embedded image from HTML: {ex.Message}");
                }
            }
        }

        return items;
    }
}
