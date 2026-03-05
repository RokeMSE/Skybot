using SkyBot.Server.Extractors;
using SkyBot.Shared.Models;

namespace SkyBot.Server.Services;

/// <summary>
/// Ingestion pipeline: extracts content from documents, chunks text,
/// optionally analyzes images with VLM, and upserts to ChromaDB.
/// Port of Python's IngestionPipeline.
/// </summary>
public class IngestionPipeline
{
    private readonly VectorDbService _vectorDb;
    private readonly ILlmService _vlmService;
    private readonly Dictionary<string, IExtractor> _extractors;
    private readonly bool _enableVlmIngestion;
    private readonly string _documentStoreDir;

    // Simple recursive character text splitter
    private readonly int _chunkSize;
    private readonly int _chunkOverlap;

    public IngestionPipeline(
        VectorDbService vectorDb,
        ILlmService vlmService,
        string imageStoreDir,
        string documentStoreDir,
        bool enableVlmIngestion = true,
        int chunkSize = 1000,
        int chunkOverlap = 100)
    {
        _vectorDb = vectorDb;
        _vlmService = vlmService;
        _enableVlmIngestion = enableVlmIngestion;
        _documentStoreDir = documentStoreDir;
        _chunkSize = chunkSize;
        _chunkOverlap = chunkOverlap;

        Directory.CreateDirectory(documentStoreDir);

        _extractors = new Dictionary<string, IExtractor>(StringComparer.OrdinalIgnoreCase)
        {
            [".pdf"] = new PdfExtractor(imageStoreDir),
            [".docx"] = new DocxExtractor(imageStoreDir),
            [".pptx"] = new PptxExtractor(imageStoreDir),
            [".xlsx"] = new XlsxExtractor(),
            [".csv"] = new CsvExtractor(),
            [".txt"] = new TextExtractor(),
            [".md"] = new TextExtractor(),
            [".log"] = new TextExtractor(),
            [".html"] = new HtmlExtractor(imageStoreDir),
            [".htm"] = new HtmlExtractor(imageStoreDir),
        };
    }

    public async Task<IngestResponse> IngestFileAsync(string filePath, string channel = "general")
    {
        var ext = Path.GetExtension(filePath).ToLower();
        if (!_extractors.ContainsKey(ext))
            throw new ArgumentException($"Unsupported file type: {ext}");

        // Copy original file to document store for serving
        var filename = Path.GetFileName(filePath);
        var docDest = Path.Combine(_documentStoreDir, filename);
        if (!File.Exists(docDest))
        {
            File.Copy(filePath, docDest, overwrite: false);
            Console.WriteLine($"Copied original file to {docDest} for serving.");
        }

        // Extract content
        var extractor = _extractors[ext];
        Console.WriteLine($"Extracting content from {filename}...");
        var items = extractor.Extract(filePath);

        var ingestId = Guid.NewGuid().ToString();
        var documents = new List<string>();
        var metadatas = new List<Dictionary<string, object>>();
        var ids = new List<string>();
        var chunkCounter = 0;

        foreach (var item in items)
        {
            if (item.Type == "text")
            {
                // Chunk text
                var chunks = SplitText(item.Content);
                foreach (var chunk in chunks)
                {
                    documents.Add(chunk);
                    var meta = new Dictionary<string, object>(item.Metadata)
                    {
                        ["source"] = item.Source,
                        ["page"] = item.PageNum,
                        ["type"] = "text",
                        ["channel"] = channel,
                        ["ingest_id"] = ingestId
                    };
                    metadatas.Add(meta);
                    ids.Add($"{ingestId}_{chunkCounter}");
                    chunkCounter++;
                }
            }
            else if (item.Type == "image" && item.ImagePath != null && File.Exists(item.ImagePath))
            {
                try
                {
                    string richContent;
                    if (_enableVlmIngestion)
                    {
                        Console.WriteLine($"Analyzing image from Page {item.PageNum} with VLM...");
                        var prompt =
                            "You are a semiconductor process engineer. Analyze this technical image. " +
                            "1. Identify the diagram type (Schematic, Cross-section, Flowchart, UI, Micrograph). " +
                            "2. Extract visible text, labels, pin numbers, and component IDs. " +
                            "3. Describe connections, material layers, or process steps shown. " +
                            "Output concise text for search indexing.";

                        using var imgStream = File.OpenRead(item.ImagePath);
                        var description = await _vlmService.AnalyzeImageAsync(imgStream, prompt);
                        richContent = $"[[IMAGE on Page {item.PageNum}]]\nDescription: {description}";
                    }
                    else
                    {
                        Console.WriteLine($"Storing image metadata from Page {item.PageNum} (VLM disabled).");
                        richContent = $"[[IMAGE on Page {item.PageNum}]]";
                    }

                    documents.Add(richContent);
                    var meta = new Dictionary<string, object>(item.Metadata)
                    {
                        ["source"] = item.Source,
                        ["page"] = item.PageNum,
                        ["type"] = "image_cad",
                        ["image_path"] = item.ImagePath,
                        ["channel"] = channel,
                        ["ingest_id"] = ingestId
                    };
                    metadatas.Add(meta);
                    ids.Add($"{ingestId}_{chunkCounter}");
                    chunkCounter++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to process image {item.ImagePath}: {ex.Message}");
                }
            }
        }

        if (documents.Count > 0)
        {
            Console.WriteLine($"Upserting {documents.Count} chunks to VectorDB...");
            await _vectorDb.UpsertAsync(documents, metadatas, ids);
        }

        return new IngestResponse
        {
            Status = "success",
            File = filename,
            Chunks = documents.Count,
            IngestId = ingestId
        };
    }

    /// <summary>
    /// Simple recursive character text splitter.
    /// Port of langchain RecursiveCharacterTextSplitter.
    /// </summary>
    private List<string> SplitText(string text)
    {
        var chunks = new List<string>();
        if (text.Length <= _chunkSize)
        {
            chunks.Add(text);
            return chunks;
        }

        var separators = new[] { "\n\n", "\n", ". ", " " };
        SplitRecursive(text, separators, 0, chunks);
        return chunks;
    }

    private void SplitRecursive(string text, string[] separators, int sepIndex, List<string> chunks)
    {
        if (text.Length <= _chunkSize)
        {
            if (!string.IsNullOrWhiteSpace(text))
                chunks.Add(text.Trim());
            return;
        }

        if (sepIndex >= separators.Length)
        {
            // Force split by chunkSize
            for (int i = 0; i < text.Length; i += _chunkSize - _chunkOverlap)
            {
                var end = Math.Min(i + _chunkSize, text.Length);
                var chunk = text.Substring(i, end - i).Trim();
                if (!string.IsNullOrWhiteSpace(chunk))
                    chunks.Add(chunk);
                if (end == text.Length) break;
            }
            return;
        }

        var sep = separators[sepIndex];
        var parts = text.Split(sep);

        var current = "";
        foreach (var part in parts)
        {
            var candidate = string.IsNullOrEmpty(current) ? part : current + sep + part;
            if (candidate.Length > _chunkSize)
            {
                if (!string.IsNullOrWhiteSpace(current))
                {
                    SplitRecursive(current, separators, sepIndex + 1, chunks);
                }
                current = part;
            }
            else
            {
                current = candidate;
            }
        }

        if (!string.IsNullOrWhiteSpace(current))
        {
            if (current.Length > _chunkSize)
                SplitRecursive(current, separators, sepIndex + 1, chunks);
            else
                chunks.Add(current.Trim());
        }
    }
}
