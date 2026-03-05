using SkyBot.Shared.Models;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Spreadsheet;

namespace SkyBot.Server.Extractors;

/// <summary>
/// Extracts text from XLSX files using OpenXml.
/// Each data row becomes its own ContentItem with headers baked in.
/// Port of Python's XLSXExtractor (openpyxl).
/// </summary>
public class XlsxExtractor : IExtractor
{
    public List<ContentItem> Extract(string filePath)
    {
        var items = new List<ContentItem>();
        var filename = Path.GetFileName(filePath);

        using var doc = SpreadsheetDocument.Open(filePath, false);
        var workbookPart = doc.WorkbookPart;
        if (workbookPart == null) return items;

        var sheets = workbookPart.Workbook.Sheets?.Elements<Sheet>().ToList() ?? new();
        var sharedStrings = workbookPart.GetPartsOfType<SharedStringTablePart>()
            .FirstOrDefault()?.SharedStringTable;

        for (int sheetIndex = 0; sheetIndex < sheets.Count; sheetIndex++)
        {
            var sheet = sheets[sheetIndex];
            var sheetName = sheet.Name?.Value ?? $"Sheet{sheetIndex + 1}";
            var relId = sheet.Id?.Value;
            if (relId == null) continue;

            var worksheetPart = (WorksheetPart)workbookPart.GetPartById(relId);
            var rows = worksheetPart.Worksheet.Descendants<Row>().ToList();

            if (rows.Count < 2)
            {
                // Only header or empty
                if (rows.Count == 1)
                {
                    var cellValues = GetRowValues(rows[0], sharedStrings)
                        .Where(v => !string.IsNullOrEmpty(v)).ToList();
                    if (cellValues.Count > 0)
                    {
                        items.Add(new ContentItem
                        {
                            Content = $"[Sheet: {sheetName}] " + string.Join(" | ", cellValues),
                            Type = "text",
                            Source = filename,
                            PageNum = sheetIndex + 1,
                            Metadata = new Dictionary<string, object> { ["sheet_name"] = sheetName }
                        });
                    }
                }
                continue;
            }

            // First row is header
            var headers = GetRowValues(rows[0], sharedStrings);

            for (int rowIndex = 1; rowIndex < rows.Count; rowIndex++)
            {
                var values = GetRowValues(rows[rowIndex], sharedStrings);
                var pairs = new List<string>();

                for (int col = 0; col < Math.Min(headers.Count, values.Count); col++)
                {
                    var val = values[col].Trim();
                    if (!string.IsNullOrEmpty(val))
                    {
                        var header = col < headers.Count ? headers[col] : $"Col{col}";
                        pairs.Add($"{header}: {val}");
                    }
                }

                if (pairs.Count > 0)
                {
                    items.Add(new ContentItem
                    {
                        Content = $"[Sheet: {sheetName}] " + string.Join(", ", pairs),
                        Type = "text",
                        Source = filename,
                        PageNum = sheetIndex + 1,
                        Metadata = new Dictionary<string, object>
                        {
                            ["sheet_name"] = sheetName,
                            ["row_index"] = rowIndex
                        }
                    });
                }
            }
        }

        return items;
    }

    private static List<string> GetRowValues(Row row, SharedStringTable? sharedStrings)
    {
        var values = new List<string>();
        foreach (var cell in row.Elements<Cell>())
        {
            var value = GetCellValue(cell, sharedStrings);
            values.Add(value);
        }
        return values;
    }

    private static string GetCellValue(Cell cell, SharedStringTable? sharedStrings)
    {
        var value = cell.CellValue?.Text ?? "";
        if (cell.DataType?.Value == CellValues.SharedString && sharedStrings != null)
        {
            if (int.TryParse(value, out var index))
            {
                var item = sharedStrings.ElementAt(index);
                return item.InnerText;
            }
        }
        return value;
    }
}
