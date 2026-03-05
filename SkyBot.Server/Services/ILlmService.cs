namespace SkyBot.Server.Services;

/// <summary>
/// Abstract interface for LLM services.
/// Supports text generation and image analysis.
/// Port of Python's VLMService + ChatService ABCs.
/// </summary>
public interface ILlmService
{
    /// <summary>
    /// Generates a text response from a prompt, optionally with a system instruction.
    /// </summary>
    Task<string> GenerateResponseAsync(string prompt, string? systemInstruction = null);

    /// <summary>
    /// Analyzes an image with a text prompt.
    /// </summary>
    Task<string> AnalyzeImageAsync(Stream imageStream, string prompt);
}
