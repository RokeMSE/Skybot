using SkyBot.Shared.Models;

namespace SkyBot.Server.Services;

/// <summary>
/// RAG Engine: retrieves context from ChromaDB and generates answers.
/// Port of Python's RAGEngine (retrieval.py).
/// </summary>
public class RagEngine
{
    private readonly VectorDbService _vectorDb;
    private readonly ILlmService _chatService;

    public RagEngine(VectorDbService vectorDb, ILlmService chatService)
    {
        _vectorDb = vectorDb;
        _chatService = chatService;
    }

    public async Task<List<string>> GetChannelsAsync()
    {
        return await _vectorDb.GetChannelsAsync();
    }

    public async Task<ChatResponse> QueryAsync(string userQuery, int nResults = 5, string? channel = null)
    {
        // 1. Retrieve — with optional channel filter
        var (documents, metadatas, ids) = await _vectorDb.QueryAsync(userQuery, nResults, channel);

        // 2. Construct Context
        var contextStr = "";
        var retrievedSources = new List<Dictionary<string, string>>();
        var imageUrls = new List<string>();
        var seenImages = new HashSet<string>();
        var schemaPages = new HashSet<(string source, string page)>();

        for (int i = 0; i < documents.Count; i++)
        {
            var doc = documents[i];
            var meta = metadatas[i];

            // Collect relevant pages for hybrid image retrieval
            if (meta.GetValueOrDefault("type") == "text")
            {
                var source = meta.GetValueOrDefault("source", "");
                var page = meta.GetValueOrDefault("page", "");
                if (!string.IsNullOrEmpty(source) && !string.IsNullOrEmpty(page))
                    schemaPages.Add((source, page));
            }

            var sourceTag = $"[Source: {meta.GetValueOrDefault("source", "Unknown")}, Page {meta.GetValueOrDefault("page", "?")}]";
            contextStr += $"\n--- {sourceTag} ---\n{doc}\n";
            retrievedSources.Add(meta);

            // Check for images (directly retrieved)
            if (meta.GetValueOrDefault("type") == "image_cad" && meta.ContainsKey("image_path"))
            {
                var imgPath = meta["image_path"];
                var imgFilename = Path.GetFileName(imgPath);
                var imgUrl = $"/static/images/{imgFilename}";

                if (seenImages.Add(imgUrl))
                    imageUrls.Add(imgUrl);
            }
        }

        // --- Hybrid Retrieval: Fetch images from relevant pages ---
        if (schemaPages.Count > 0)
        {
            Console.WriteLine($"Hybrid Retrieval: Checking for images on {schemaPages.Count} pages...");
            try
            {
                foreach (var (source, page) in schemaPages)
                {
                    var whereFilter = new Dictionary<string, object>
                    {
                        ["$and"] = new[]
                        {
                            new Dictionary<string, object> { ["type"] = "image_cad" },
                            new Dictionary<string, object> { ["source"] = source },
                            new Dictionary<string, object> { ["page"] = page }
                        }
                    };

                    var (imgMetas, _) = await _vectorDb.GetByFilterAsync(whereFilter);

                    foreach (var meta in imgMetas)
                    {
                        if (meta.ContainsKey("image_path"))
                        {
                            var imgFilename = Path.GetFileName(meta["image_path"]);
                            var imgUrl = $"/static/images/{imgFilename}";

                            if (seenImages.Add(imgUrl))
                            {
                                Console.WriteLine($"Hybrid Retrieval: Found related image {imgFilename}");
                                imageUrls.Insert(0, imgUrl);
                                if (!retrievedSources.Contains(meta))
                                    retrievedSources.Add(meta);
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Hybrid retrieval error: {ex.Message}");
            }
        }

        if (string.IsNullOrWhiteSpace(contextStr))
        {
            return new ChatResponse
            {
                Answer = "I couldn't find any relevant information in the uploaded documents to answer your question.",
                Citations = new List<Citation>(),
                Images = new List<string>()
            };
        }

        // 3. Build source-to-document-URL mapping for hyperlinks
        var sourceDocLinks = new Dictionary<string, Dictionary<string, string>>();
        foreach (var meta in retrievedSources)
        {
            var source = meta.GetValueOrDefault("source", "");
            var page = meta.GetValueOrDefault("page", "");
            if (!string.IsNullOrEmpty(source) && !string.IsNullOrEmpty(page))
            {
                if (!sourceDocLinks.ContainsKey(source))
                    sourceDocLinks[source] = new Dictionary<string, string>();
                sourceDocLinks[source][page] = $"/static/documents/{source}#page={page}";
            }
        }

        // 4. Build page-to-image-URL mapping for inline embeds
        var pageToImageMap = new Dictionary<string, string>();
        foreach (var meta in retrievedSources)
        {
            if (meta.GetValueOrDefault("type") == "image_cad"
                && meta.ContainsKey("image_path")
                && meta.ContainsKey("page"))
            {
                var key = $"{meta.GetValueOrDefault("source", "")}_p{meta["page"]}";
                var imgFilename = Path.GetFileName(meta["image_path"]);
                pageToImageMap[key] = $"/static/images/{imgFilename}";
            }
        }

        // 5. Generate Answer
        var systemInstruction =
            "You are an expert Semiconductor Manufacturing Assistant. " +
            "You are provided with text context (which may contain pre-generated image descriptions) and actual images. " +
            "CRITICAL: Prioritize your own visual analysis of the provided images over the pre-generated text descriptions if they conflict. " +
            "Answer the user's question based on the provided context (text and images). " +
            "If the context does not contain relevant information, say so clearly. " +
            "Cite the page numbers provided in the context.\n\n" +
            "--- IMAGE DISPLAY RULES ---\n" +
            "When an extracted image (JPEG/PNG) is available for a page, EMBED it inline using markdown:\n" +
            "  ![Description of diagram](/static/images/image_filename.png)\n\n" +
            $"Available image URLs by source and page:\n{System.Text.Json.JsonSerializer.Serialize(pageToImageMap)}\n\n" +
            "If the context references a chart, graph, or diagram that does NOT have an extracted image URL above,\n" +
            "then hyperlink to the source document page instead:\n" +
            "  📄 [FileName — Page X](/static/documents/FileName#page=X)\n\n" +
            $"Available document links:\n{System.Text.Json.JsonSerializer.Serialize(sourceDocLinks)}\n\n" +
            "You MAY freely describe or discuss the content of images and diagrams.\n" +
            "You MUST NOT generate, draw, or recreate charts, graphs, or diagrams using markdown, ASCII art, or code blocks.\n" +
            "--------------------------";

        var finalPrompt = $"USER QUESTION: {userQuery}\n\nCONTEXT DATA:\n{contextStr}";

        var answer = await _chatService.GenerateResponseAsync(finalPrompt, systemInstruction);

        // Build citations
        var citations = retrievedSources.Take(3).Select(m => new Citation
        {
            Source = m.GetValueOrDefault("source", "Unknown"),
            Page = m.GetValueOrDefault("page", "?"),
            Type = m.GetValueOrDefault("type")
        }).ToList();

        return new ChatResponse
        {
            Answer = answer,
            Citations = citations,
            Images = imageUrls.Take(3).ToList()
        };
    }
}
