using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SkyBot.Blazor;
using SkyBot.Blazor.Services;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

// Configure HttpClient to point at the .NET backend
builder.Services.AddScoped(sp => new HttpClient
{
    BaseAddress = new Uri("http://localhost:5000"),
    Timeout = TimeSpan.FromMinutes(10) // Ingestion can take a while
});

builder.Services.AddScoped<SkybotApiService>();

await builder.Build().RunAsync();
