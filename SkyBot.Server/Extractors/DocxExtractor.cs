using SkyBot.Shared.Models;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using SixLabors.ImageSharp;

namespace SkyBot.Server.Extractors;

/// <summary>
/// Extracts text and images from DOCX files using OpenXml.
/// Port of Python's DOCXExtractor (python-docx).
/// </summary>
public class DocxExtractor : IExtractor
{
    private readonly string _imageStoreDir;

    public DocxExtractor(string imageStoreDir)
    {
        _imageStoreDir = imageStoreDir;
        Directory.CreateDirectory(_imageStoreDir);
    }

    public List<ContentItem> Extract(string filePath)
    {
        var items = new List<ContentItem>();
        var filename = Path.GetFileName(filePath);

        using var doc = WordprocessingDocument.Open(filePath, false);
        var body = doc.MainDocumentPart?.Document.Body;
        if (body == null) return items;

        // --- 1. Text Extraction ---
        var paragraphs = body.Elements<Paragraph>()
            .Select(p => p.InnerText.Trim())
            .Where(t => !string.IsNullOrEmpty(t))
            .ToList();

        if (paragraphs.Count > 0)
        {
            items.Add(new ContentItem
            {
                Content = string.Join("\n", paragraphs),
                Type = "text",
                Source = filename,
                PageNum = 1 // DOCX doesn't have fixed pages
            });
        }

        // --- 2. Image Extraction ---
        var mainPart = doc.MainDocumentPart;
        if (mainPart != null)
        {
            foreach (var imagePart in mainPart.ImageParts)
            {
                try
                {
                    using var stream = imagePart.GetStream();
                    using var img = SixLabors.ImageSharp.Image.Load(stream);

                    if (img.Width < 100 || img.Height < 100)
                        continue;

                    var imageId = $"{Guid.NewGuid()}.png";
                    var savePath = Path.Combine(_imageStoreDir, imageId);
                    img.SaveAsPng(savePath);

                    items.Add(new ContentItem
                    {
                        Content = "[[Image extracted from DOCX]]",
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
                    Console.WriteLine($"Failed to extract image from DOCX: {ex.Message}");
                }
            }
        }

        return items;
    }
}
