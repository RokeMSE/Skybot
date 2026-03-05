FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY *.sln ./
COPY SkyBot.Server/SkyBot.Server.csproj SkyBot.Server/
COPY SkyBot.Shared/SkyBot.Shared.csproj SkyBot.Shared/
COPY SkyBot.Blazor/SkyBot.Blazor.csproj SkyBot.Blazor/

RUN dotnet restore

COPY . .
WORKDIR /src/SkyBot.Server
RUN dotnet build "SkyBot.Server.csproj" -c Release -o /app/build

FROM build AS publish
WORKDIR /src/SkyBot.Server
RUN dotnet publish "SkyBot.Server.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "SkyBot.Server.dll"]