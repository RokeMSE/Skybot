using Azure.AI.OpenAI;
using OpenAI.Chat;

namespace SkyBot.Server.Services;

/// <summary>
/// OpenAI/Azure OpenAI LLM service using Azure.AI.OpenAI SDK.
/// Port of Python's OpenAIService.
/// </summary>
public class OpenAiLlmService : ILlmService
{
    private readonly ChatClient _chatClient;

    public OpenAiLlmService(string apiKey, string model, string? endpoint = null)
    {
        if (!string.IsNullOrEmpty(endpoint))
        {
            var azureClient = new AzureOpenAIClient(
                new Uri(endpoint),
                new System.ClientModel.ApiKeyCredential(apiKey));
            _chatClient = azureClient.GetChatClient(model);
        }
        else
        {
            var client = new OpenAI.OpenAIClient(new System.ClientModel.ApiKeyCredential(apiKey));
            _chatClient = client.GetChatClient(model);
        }
    }

    public async Task<string> GenerateResponseAsync(string prompt, string? systemInstruction = null)
    {
        try
        {
            var messages = new List<ChatMessage>();

            if (!string.IsNullOrEmpty(systemInstruction))
                messages.Add(ChatMessage.CreateSystemMessage(systemInstruction));

            messages.Add(ChatMessage.CreateUserMessage(prompt));

            var options = new ChatCompletionOptions { Temperature = 0.5f };
            var result = await _chatClient.CompleteChatAsync(messages, options);

            return result.Value.Content[0].Text ?? "No response generated.";
        }
        catch (Exception ex)
        {
            return $"Error generating response with OpenAI: {ex.Message}";
        }
    }

    public async Task<string> AnalyzeImageAsync(Stream imageStream, string prompt)
    {
        try
        {
            using var ms = new MemoryStream();
            await imageStream.CopyToAsync(ms);
            var base64 = Convert.ToBase64String(ms.ToArray());
            var dataUrl = $"data:image/png;base64,{base64}";

            var imagePart = ChatMessageContentPart.CreateImagePart(new Uri(dataUrl));
            var textPart = ChatMessageContentPart.CreateTextPart(prompt);

            var messages = new List<ChatMessage>
            {
                ChatMessage.CreateUserMessage(textPart, imagePart)
            };

            var options = new ChatCompletionOptions { Temperature = 0.5f };
            var result = await _chatClient.CompleteChatAsync(messages, options);

            return result.Value.Content[0].Text ?? "No analysis generated.";
        }
        catch (Exception ex)
        {
            return $"Error analyzing image with OpenAI: {ex.Message}";
        }
    }
}
