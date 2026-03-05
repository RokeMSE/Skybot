using SkyBot.Shared.Models;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Presentation;
using SixLabors.ImageSharp;

namespace SkyBot.Server.Extractors;

/// <summary>
/// Extracts text and images from PPTX files using OpenXml.
/// Port of Python's PPTXExtractor (python-pptx).
/// </summary>
public class PptxExtractor : IExtractor
{
    private readonly string _imageStoreDir;

    public PptxExtractor(string imageStoreDir)
    {
        _imageStoreDir = imageStoreDir;
        Directory.CreateDirectory(_imageStoreDir);
    }

    public List<ContentItem> Extract(string filePath)
    {
        var items = new List<ContentItem>();
        var filename = Path.GetFileName(filePath);

        using var prs = PresentationDocument.Open(filePath, false);
        var presentationPart = prs.PresentationPart;
        if (presentationPart?.Presentation.SlideIdList == null) return items;

        var slideIds = presentationPart.Presentation.SlideIdList.ChildElements
            .OfType<SlideId>().ToList();

        for (int slideIndex = 0; slideIndex < slideIds.Count; slideIndex++)
        {
            var pageNum = slideIndex + 1;
            var slideId = slideIds[slideIndex];
            var relId = slideId.RelationshipId?.Value;
            if (relId == null) continue;

            var slidePart = (SlidePart)presentationPart.GetPartById(relId);
            var slideTexts = new List<string>();

            // --- 1. Text Extraction ---
            foreach (var textBody in slidePart.Slide.Descendants<DocumentFormat.OpenXml.Drawing.Text>())
            {
                var text = textBody.Text?.Trim();
                if (!string.IsNullOrEmpty(text))
                    slideTexts.Add(text);
            }

            if (slideTexts.Count > 0)
            {
                items.Add(new ContentItem
                {
                    Content = string.Join("\n", slideTexts),
                    Type = "text",
                    Source = filename,
                    PageNum = pageNum
                });
            }

            // --- 2. Image Extraction ---
            foreach (var imagePart in slidePart.ImageParts)
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
                        Content = $"[[Image extracted from Slide {pageNum}]]",
                        Type = "image",
                        Source = filename,
                        PageNum = pageNum,
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
                    Console.WriteLine($"Failed to extract image on slide {pageNum}: {ex.Message}");
                }
            }
        }

        return items;
    }
}
