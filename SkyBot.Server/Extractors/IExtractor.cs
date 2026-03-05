using SkyBot.Shared.Models;

namespace SkyBot.Server.Extractors;

/// <summary>
/// Abstract interface for document extractors.
/// Each extractor handles a specific file type.
/// </summary>
public interface IExtractor
{
    List<ContentItem> Extract(string filePath);
}
