# Place this script in the root of your project (e.g., D:\Documents\cyber\projects\Sentient-New\Code)

<#
.SYNOPSIS
    Starts all backend services, workers, and the frontend client for the Sentient project.

.DESCRIPTION
    This script automates the startup of all necessary services for local development.
    It launches each service in its own dedicated PowerShell terminal window with a clear title.

    The script handles:
    - Starting databases like MongoDB (as admin) and Docker services (Waha, PGVector, Chroma, LiteLLM).
    - Launching the Redis message broker within the Windows Subsystem for Linux (WSL).
    - Dynamically discovering and starting all MCP (Modular Companion Protocol) servers.
    - Activating the Python virtual environment for all backend scripts.
    - Running the Celery worker and beat scheduler for background tasks.
    - Starting the main FastAPI server and the Next.js frontend client.

.NOTES
    - Run this script from your project's root directory.
    - Requires Docker Desktop to be installed and running.
    - You may need to adjust your PowerShell execution policy to run this script.
      Open PowerShell as an Administrator and run:
      Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#>

# --- Configuration ---
# Please review and update these paths to match your local setup.

# The name of your WSL distribution where Redis is installed.
$wslDistroName = "Ubuntu"

# --- Script Body ---
try {
    # Prerequisite check
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker command not found. Please install Docker Desktop and ensure it is running."
    }

    # Get the directory where the script is located (your project root)
    $projectRoot = $PSScriptRoot
    if (-not $projectRoot) { $projectRoot = Get-Location }

    # Define key paths
    $srcPath = Join-Path -Path $projectRoot -ChildPath "src"
    $serverPath = Join-Path -Path $srcPath -ChildPath "server"
    $clientPath = Join-Path -Path $srcPath -ChildPath "client"
    $mcpHubPath = Join-Path -Path $serverPath -ChildPath "mcp_hub"
    $venvActivatePath = Join-Path -Path $serverPath -ChildPath "venv\Scripts\activate.ps1"

    # --- Path Validation ---
    if (-not (Test-Path -Path $srcPath)) { throw "The 'src' directory was not found. Please run this script from the project root." }
    if (-not (Test-Path -Path $serverPath)) { throw "The 'src/server' directory was not found." }
    if (-not (Test-Path -Path $clientPath)) { throw "The 'src/client' directory was not found." }
    if (-not (Test-Path -Path $mcpHubPath)) { throw "The 'src/server/mcp_hub' directory was not found." }
    if (-not (Test-Path -Path $venvActivatePath)) { throw "The venv activation script was not found at '$venvActivatePath'." }

    $envFilePath = Join-Path -Path $serverPath -ChildPath ".env"
    $redisPassword = ""
    if (Test-Path $envFilePath) {
        $envContent = Get-Content $envFilePath
        $passwordLine = $envContent | Select-String -Pattern "^\s*REDIS_PASSWORD\s*=\s*(.+)"
        if ($passwordLine) {
            $redisPassword = $passwordLine.Matches[0].Groups[1].Value.Trim()
        }
    }
    if (-not $redisPassword) {
        throw "Could not find REDIS_PASSWORD in the src/server/.env file."
    }

    # Helper function to start a process in a new terminal window
    function Start-NewTerminal {
        param(
            [string]$WindowTitle,
            [string]$Command,
            [string]$WorkDir = $projectRoot,
            [switch]$NoExit = $true
        )
        # Using -NoExit keeps the window open to see output/errors
        $psCommand = "Set-Location -Path '$WorkDir'; `$Host.UI.RawUI.WindowTitle = '$WindowTitle'; $Command"
        $startArgs = @{
            FilePath         = "powershell.exe"
            WorkingDirectory = $WorkDir
            ArgumentList     = "-NoExit", "-Command", $psCommand
        }
        if (-not $NoExit) {
            # For fire-and-forget commands
            $startArgs.ArgumentList = "-Command", $psCommand
        }
        Start-Process @startArgs
    }

    # --- 1. Start Databases & Core Infrastructure ---
    Write-Host "`n--- 1. Starting Databases & Core Infrastructure ---" -ForegroundColor Cyan

    # Start MongoDB Service (requires admin)
    Write-Host "🚀 Launching MongoDB Service (requires admin)..." -ForegroundColor Yellow
    Start-Process powershell.exe -Verb RunAs -ArgumentList "Start-Service -Name 'MongoDB' -ErrorAction SilentlyContinue; if (`$?) { Write-Host 'MongoDB service started successfully.' -ForegroundColor Green } else { Write-Host 'MongoDB service was already running or failed to start.' -ForegroundColor Yellow }; Read-Host 'Press Enter to close this admin window.'"
    Start-Sleep -Seconds 3

    
    # (Redis is now started via Docker below)

    
    # Start Docker Containers (Waha, PGVector, Chroma, LiteLLM)
    Write-Host "🚀 Launching Docker services (Waha, PGVector, Chroma, LiteLLM)..." -ForegroundColor Yellow
    $dockerServices = @(
        @{ Name = "WAHA"; File = "start_waha.yaml" },
        @{ Name = "Redis"; File = "start_redis.yaml" },
        @{ Name = "PGVector"; File = "start_pgvector.yaml" },
        @{ Name = "ChromaDB"; File = "start_chroma.yaml" },
        @{ Name = "LiteLLM"; File = "start_litellm.yaml" }
    )

    foreach ($service in $dockerServices) {
        $composeFile = Join-Path -Path $projectRoot -ChildPath $service.File
        if (Test-Path $composeFile) {
            Write-Host "   - Starting $($service.Name) from '$($service.File)'..." -ForegroundColor Gray
            docker compose -f $composeFile up -d
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "   - Command to start $($service.Name) failed. Check Docker's output above."
            }
            else {
                Write-Host "   - $($service.Name) start command issued successfully." -ForegroundColor Green
            }
        }
        else {
            Write-Warning "   - Docker compose file not found: '$($service.File)'. Skipping."
        }
    }
    
    Write-Host "Waiting a few seconds for Docker containers to initialize..."
    Start-Sleep -Seconds 5


    # --- 2. Start MCP Servers ---
    Write-Host "`n--- 2. Starting All MCP Servers ---" -ForegroundColor Cyan
    $mcpServers = Get-ChildItem -Path $mcpHubPath -Directory | Select-Object -ExpandProperty Name
    if ($mcpServers.Count -eq 0) { throw "No MCP server directories found in '$mcpHubPath'." }

    Write-Host "Found the following MCP servers to start:" -ForegroundColor Green
    $mcpServers | ForEach-Object { Write-Host " - $_" }
    Write-Host ""

    foreach ($serverName in $mcpServers) {
        $windowTitle = "MCP - $($serverName.ToUpper())"
        $pythonModule = "mcp_hub.$serverName.main"
        $commandToRun = "& '$venvActivatePath'; python -m $pythonModule"
        Write-Host "🚀 Launching $windowTitle..." -ForegroundColor Yellow
        Start-NewTerminal -WindowTitle $windowTitle -Command $commandToRun -WorkDir $serverPath
        Start-Sleep -Milliseconds 500
    }

    # --- 3. Resetting Queues & State (for clean development starts) ---
    Write-Host "`n--- 3. Resetting Queues & State ---" -ForegroundColor Cyan

    # Clear Redis (Celery queue) - This is fast, runs in the current window.
    Write-Host "🚀 Flushing Redis database (Celery Queue)..." -ForegroundColor Yellow
    $redisFlushCommand = "wsl -d $wslDistroName -e redis-cli FLUSHALL"
    Start-NewTerminal -WindowTitle "RESET - Redis Flush" -Command $redisFlushCommand -WorkDir $serverPath -NoExit:$false

    # --- 4. Start Backend Workers ---
    Write-Host "`n--- 4. Starting Backend Workers ---" -ForegroundColor Cyan
    $workerServices = @(
        @{ Name = "Celery Worker"; Command = "& '$venvActivatePath'; celery -A workers.celery_app worker --loglevel=info --pool=solo" },
        @{ Name = "Celery Beat Scheduler"; Command = "& '$venvActivatePath'; celery -A workers.celery_app beat --loglevel=info" }
    )

    foreach ($service in $workerServices) {
        $windowTitle = "WORKER - $($service.Name)"
        Write-Host "🚀 Launching $windowTitle..." -ForegroundColor Yellow
        Start-NewTerminal -WindowTitle $windowTitle -Command $service.Command -WorkDir $serverPath
        Start-Sleep -Milliseconds 500
    }

    # --- 5. Start Main API Server and Frontend Client ---
    Write-Host "`n--- 5. Starting Main API and Client ---" -ForegroundColor Cyan

    # Start Main FastAPI Server
    Write-Host "🚀 Launching Main API Server..." -ForegroundColor Yellow
    $mainApiCommand = "& '$venvActivatePath'; python -m main.app"
    Start-NewTerminal -WindowTitle "API - Main Server" -Command $mainApiCommand -WorkDir $serverPath
    Start-Sleep -Seconds 3

    # Start Next.js Client
    Write-Host "🚀 Launching Next.js Client..." -ForegroundColor Yellow
    Start-NewTerminal -WindowTitle "CLIENT - Next.js" -Command "npm run dev" -WorkDir $clientPath

    Write-Host "`n✅ All services have been launched successfully in new terminals." -ForegroundColor Green
}
catch {
    Write-Error "An error occurred during startup: $_"
    Read-Host "Press Enter to exit..."
}