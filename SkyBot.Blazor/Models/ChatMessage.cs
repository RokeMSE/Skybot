namespace SkyBot.Blazor.Models;

public enum MessageType { User, System }

public class ChatMessage
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N")[..8];
    public string Content { get; set; } = string.Empty;
    public MessageType Type { get; set; }
    public bool IsLoading { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.Now;
}
