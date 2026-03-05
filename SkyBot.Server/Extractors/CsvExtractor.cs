using SkyBot.Shared.Models;
using Microsoft.VisualBasic.FileIO;

namespace SkyBot.Server.Extractors;

/// <summary>
/// Extracts text from CSV files.
/// Each row becomes its own ContentItem with headers baked in.
/// </summary>
public class CsvExtractor : IExtractor
{
    public List<ContentItem> Extract(string filePath)
    {
        var items = new List<ContentItem>();
        var filename = Path.GetFileName(filePath);

        var rows = new List<string[]>();
        using (var parser = new TextFieldParser(filePath))
        {
            parser.TextFieldType = FieldType.Delimited;
            parser.SetDelimiters(",");
            parser.HasFieldsEnclosedInQuotes = true;

            while (!parser.EndOfData)
            {
                var fields = parser.ReadFields();
                if (fields != null)
                    rows.Add(fields);
            }
        }

        if (rows.Count < 2)
        {
            if (rows.Count == 1)
            {
                items.Add(new ContentItem
                {
                    Content = string.Join(" | ", rows[0]),
                    Type = "text",
                    Source = filename,
                    PageNum = 1
                });
            }
            return items;
        }

        var headers = rows[0];

        for (int rowIndex = 1; rowIndex < rows.Count; rowIndex++)
        {
            var row = rows[rowIndex];
            var pairs = new List<string>();

            for (int col = 0; col < Math.Min(headers.Length, row.Length); col++)
            {
                var val = row[col].Trim();
                if (!string.IsNullOrEmpty(val))
                    pairs.Add($"{headers[col]}: {val}");
            }

            if (pairs.Count > 0)
            {
                items.Add(new ContentItem
                {
                    Content = string.Join(", ", pairs),
                    Type = "text",
                    Source = filename,
                    PageNum = rowIndex,
                    Metadata = new Dictionary<string, object> { ["row_index"] = rowIndex }
                });
            }
        }

        return items;
    }
}
