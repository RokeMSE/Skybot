using SkyBot.Shared.Models;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;
using SixLabors.ImageSharp;

namespace SkyBot.Server.Extractors;

/// <summary>
/// Extracts text and images from PDF files using PdfPig.
/// Handles PNG, JPEG, and raw image byte extraction.
/// </summary>
public class PdfExtractor : IExtractor
{
    private readonly string _imageStoreDir;

    public PdfExtractor(string imageStoreDir)
    {
        _imageStoreDir = imageStoreDir;
        Directory.CreateDirectory(_imageStoreDir);
    }

    public List<ContentItem> Extract(string filePath)
    {
        var items = new List<ContentItem>();
        var filename = Path.GetFileName(filePath);

        using var document = PdfDocument.Open(filePath);

        foreach (var page in document.GetPages())
        {
            var pageNum = page.Number;

            // --- 1. Text Extraction ---
            var text = page.Text;
            if (!string.IsNullOrWhiteSpace(text))
            {
                items.Add(new ContentItem
                {
                    Content = text,
                    Type = "text",
                    Source = filename,
                    PageNum = pageNum
                });
            }

            // --- 2. Image Extraction ---
            try
            {
                foreach (var image in page.GetImages())
                {
                    try
                    {
                        byte[]? imageBytes = null;
                        string extension = "png";

                        // Strategy 1: Try TryGetPng (handles many formats)
                        if (image.TryGetPng(out var pngBytes) && pngBytes != null && pngBytes.Length > 100)
                        {
                            imageBytes = pngBytes;
                            extension = "png";
                        }
                        // Strategy 2: Try raw bytes (works for JPEG/DCTDecode images)
                        else
                        {
                            var rawBytes = image.RawBytes.ToArray();
                            if (rawBytes.Length > 100)
                            {
                                // Check for JPEG header (FFD8FF)
                                if (rawBytes.Length >= 3 &&
                                    rawBytes[0] == 0xFF && rawBytes[1] == 0xD8 && rawBytes[2] == 0xFF)
                                {
                                    imageBytes = rawBytes;
                                    extension = "jpg";
                                }
                                // Check for PNG header (89504E47)
                                else if (rawBytes.Length >= 4 &&
                                         rawBytes[0] == 0x89 && rawBytes[1] == 0x50 &&
                                         rawBytes[2] == 0x4E && rawBytes[3] == 0x47)
                                {
                                    imageBytes = rawBytes;
                                    extension = "png";
                                }
                                // Try to load raw bytes with ImageSharp (handles many formats)
                                else
                                {
                                    try
                                    {
                                        using var testStream = new MemoryStream(rawBytes);
                                        using var testImg = SixLabors.ImageSharp.Image.Load(testStream);
                                        // If we got here, ImageSharp can read it
                                        imageBytes = rawBytes;
                                        extension = "png"; // will re-save as PNG
                                    }
                                    catch
                                    {
                                        // Can't decode, skip
                                    }
                                }
                            }
                        }

                        if (imageBytes == null || imageBytes.Length < 100)
                            continue;

                        // Load with ImageSharp to validate and get dimensions
                        using var memStream = new MemoryStream(imageBytes);
                        SixLabors.ImageSharp.Image img;
                        try
                        {
                            img = SixLabors.ImageSharp.Image.Load(memStream);
                        }
                        catch
                        {
                            // If ImageSharp can't load it, try saving raw bytes directly
                            if (extension == "jpg")
                            {
                                var imageId = $"{Guid.NewGuid()}.jpg";
                                var savePath = Path.Combine(_imageStoreDir, imageId);
                                File.WriteAllBytes(savePath, imageBytes);
                                Console.WriteLine($"  Saved raw JPEG from page {pageNum}: {imageId}");

                                items.Add(new ContentItem
                                {
                                    Content = $"[[Image extracted from Page {pageNum}]]",
                                    Type = "image",
                                    Source = filename,
                                    PageNum = pageNum,
                                    ImagePath = savePath,
                                    Metadata = new Dictionary<string, object>
                                    {
                                        ["width"] = 0,
                                        ["height"] = 0,
                                        ["format"] = "jpeg_raw"
                                    }
                                });
                            }
                            continue;
                        }

                        using (img)
                        {
                            // Filter tiny images (icons, bullets, decorations)
                            if (img.Width < 100 || img.Height < 100)
                                continue;

                            var imageId = $"{Guid.NewGuid()}.png";
                            var savePath = Path.Combine(_imageStoreDir, imageId);
                            img.SaveAsPng(savePath);
                            Console.WriteLine($"  Extracted image from page {pageNum}: {img.Width}x{img.Height} -> {imageId}");

                            items.Add(new ContentItem
                            {
                                Content = $"[[Image extracted from Page {pageNum}]]",
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
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Failed to extract image on page {pageNum}: {ex.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Image extraction error on page {pageNum}: {ex.Message}");
            }
        }

        Console.WriteLine($"PDF extraction complete: {items.Count(i => i.Type == "text")} text chunks, {items.Count(i => i.Type == "image")} images");
        return items;
    }
}
