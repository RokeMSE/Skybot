using SkyBot.Server.Services;
using SkyBot.Shared.Models;
using Microsoft.Extensions.FileProviders;

var builder = WebApplication.CreateBuilder(args);

// --- Configuration ---
var config = builder.Configuration;
var llmProvider = config["Llm:Provider"] ?? "gemini";

// Directories
var imageStoreDir = Path.Combine(Directory.GetCurrentDirectory(), "static", "images");
var documentStoreDir = Path.Combine(Directory.GetCurrentDirectory(), "static", "documents");
var uploadDir = Path.Combine(Directory.GetCurrentDirectory(), "uploads");
Directory.CreateDirectory(imageStoreDir);
Directory.CreateDirectory(documentStoreDir);
Directory.CreateDirectory(uploadDir);

// --- Register Services ---

// 1. Ollama Embedder (always needed for ChromaDB embeddings)
var ollamaEmbedder = new OllamaLlmService(
    model: config["Llm:Ollama:Model"] ?? "qwen3-vl:4b",
    endpoint: config["Llm:Ollama:Endpoint"] ?? "http://localhost:11434"
);

// 2. VectorDb
var vectorDb = new VectorDbService(
    chromaEndpoint: config["ChromaDb:Endpoint"] ?? "http://localhost:8100",
    collectionName: config["ChromaDb:CollectionName"] ?? "semicon_knowledge_base",
    ollamaEmbedder: ollamaEmbedder
);
builder.Services.AddSingleton(vectorDb);

// 3. LLM Service (based on configured provider)
ILlmService llmService = llmProvider switch
{
    "gemini" => new GeminiLlmService(
        apiKey: config["Llm:Gemini:ApiKey"] ?? "",
        model: config["Llm:Gemini:Model"] ?? "gemini-2.5-flash",
        baseUrl: config["Llm:Gemini:Endpoint"]
    ),
    "openai" => new OpenAiLlmService(
        apiKey: config["Llm:OpenAI:ApiKey"] ?? "",
        model: config["Llm:OpenAI:Model"] ?? "gpt-4o",
        endpoint: config["Llm:OpenAI:Endpoint"]
    ),
    "ollama" => ollamaEmbedder, // reuse the same Ollama instance
    _ => throw new InvalidOperationException($"Unknown LLM provider: {llmProvider}")
};
builder.Services.AddSingleton(llmService);

// 4. RAG Engine
var ragEngine = new RagEngine(vectorDb, llmService);
builder.Services.AddSingleton(ragEngine);

// 5. Ingestion Pipeline
var enableVlm = config.GetValue("EnableVlmIngestion", true);
var ingestionPipeline = new IngestionPipeline(
    vectorDb: vectorDb,
    vlmService: llmService,
    imageStoreDir: imageStoreDir,
    documentStoreDir: documentStoreDir,
    enableVlmIngestion: enableVlm
);
builder.Services.AddSingleton(ingestionPipeline);

// --- CORS ---
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

var app = builder.Build();
app.UseCors();

// --- Static Files ---
app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(
        Path.Combine(Directory.GetCurrentDirectory(), "static")),
    RequestPath = "/static"
});

// --- API Endpoints ---

// POST /chat
app.MapPost("/chat", async (ChatRequest request, RagEngine rag) =>
{
    try
    {
        var response = await rag.QueryAsync(request.Query, channel: request.Channel);
        return Results.Ok(response);
    }
    catch (Exception ex)
    {
        return Results.Problem($"Chat failed: {ex.Message}", statusCode: 500);
    }
});

// POST /ingest
app.MapPost("/ingest", async (HttpRequest request, IngestionPipeline pipeline) =>
{
    try
    {
        if (!request.HasFormContentType)
            return Results.BadRequest("Expected multipart form data.");

        var form = await request.ReadFormAsync();
        var file = form.Files.GetFile("file");
        var channel = form["channel"].FirstOrDefault() ?? "general";

        if (file == null)
            return Results.BadRequest("No file provided.");

        // Save uploaded file to disk
        var filePath = Path.Combine(uploadDir, file.FileName);
        using (var stream = File.Create(filePath))
        {
            await file.CopyToAsync(stream);
        }

        var result = await pipeline.IngestFileAsync(filePath, channel);
        return Results.Ok(result);
    }
    catch (Exception ex)
    {
        return Results.Problem($"Ingestion failed: {ex.Message}", statusCode: 500);
    }
});

// GET /channels
app.MapGet("/channels", async (RagEngine rag) =>
{
    try
    {
        var channels = await rag.GetChannelsAsync();
        return Results.Ok(new ChannelList { Channels = channels });
    }
    catch (Exception ex)
    {
        return Results.Problem($"Failed to fetch channels: {ex.Message}", statusCode: 500);
    }
});

Console.WriteLine($"=== Skybot .NET Backend ===");
Console.WriteLine($"LLM Provider: {llmProvider}");
Console.WriteLine($"ChromaDB: {config["ChromaDb:Endpoint"] ?? "http://localhost:8100"}");
Console.WriteLine($"VLM Ingestion: {enableVlm}");

app.Run();
