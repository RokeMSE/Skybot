# Skybot

```
SkyBot.Blazor/
├── Models/
│   ├── ChatMessage.cs       ← UI state model
│   ├── ChatRequest.cs       ← /chat request DTO
│   ├── ChatResponse.cs      ← /chat response with citations + images
│   ├── ChannelList.cs        ← /channels response DTO
│   └── IngestResponse.cs    ← /ingest response DTO
├── Services/
│   └── SkybotApiService.cs  ← HttpClient wrapper (3 endpoints)
├── Components/
│   ├── Sidebar.razor         ← File upload + channel management
│   ├── ChatHistory.razor     ← Message list with auto-scroll
│   ├── ChatInput.razor       ← Textarea + send button
│   └── MessageBubble.razor   ← Markdig markdown rendering
├── Layout/
│   └── MainLayout.razor      ← App shell
├── Pages/
│   └── Index.razor           ← Main page (orchestrates everything)
├── wwwroot/
│   ├── css/app.css           ← Ported CSS with animations
│   └── index.html            ← Blazor host page
├── Program.cs                ← DI setup, HttpClient → localhost:8000
└── App.razor                 ← Router
```