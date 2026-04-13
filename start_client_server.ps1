<#
.SYNOPSIS
    Starts the frontend (Next.js) and backend (FastAPI) services for the Sentient project.

.DESCRIPTION
    This script runs:
    - The FastAPI backend server
    - The Next.js frontend client

    Both will be launched in separate PowerShell terminal windows with appropriate titles.

.NOTES
    - Run this from your project's root directory.
    - Next.js runs from src/client (npm install / npm run dev there).
#>

# --- Configuration ---
$projectRoot = $PSScriptRoot
if (-not $projectRoot) { $projectRoot = Get-Location }

$srcPath = Join-Path $projectRoot "src"
$serverPath = Join-Path $srcPath "server"
$clientPath = Join-Path $srcPath "client"
$venvActivatePath = Join-Path $serverPath "venv\Scripts\activate.ps1"
$clientPackageJson = Join-Path $clientPath "package.json"

# --- Validation ---
if (-not (Test-Path $clientPath)) { throw "Frontend directory 'src/client' not found." }
if (-not (Test-Path -LiteralPath $clientPackageJson)) { throw "Missing src/client/package.json — run npm install in src/client." }
if (-not (Test-Path $venvActivatePath)) { throw "Virtual environment activation script not found at '$venvActivatePath'." }

$serverPath = (Get-Item -LiteralPath $serverPath).FullName
$clientPath = (Get-Item -LiteralPath $clientPath).FullName

Write-Host "✅ Paths verified (API: $serverPath | Next.js: $clientPath)" -ForegroundColor Green

# --- Helper Function ---
function Start-NewTerminal {
    param (
        [string]$WindowTitle,
        [string]$Command,
        [string]$WorkDir = $projectRoot
    )
    $wd = (Get-Item -LiteralPath $WorkDir).FullName
    $psCommand = "`$Host.UI.RawUI.WindowTitle = '$WindowTitle'; $Command"
    Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $psCommand -WorkingDirectory $wd
}

# --- Start Backend ---
Write-Host "🚀 Launching FastAPI Backend..." -ForegroundColor Yellow
$backendCommand = "& '$venvActivatePath'; python -m main.app"
Start-NewTerminal -WindowTitle "API - Main Server" -Command $backendCommand -WorkDir $serverPath

# --- Start Frontend (cwd = src/client) ---
Write-Host "🚀 Launching Next.js Frontend (src\client)..." -ForegroundColor Yellow
Start-NewTerminal -WindowTitle "CLIENT - Next.js (src\client)" -Command "npm run dev" -WorkDir $clientPath

Write-Host "`n✅ Frontend and backend launched successfully in new terminals." -ForegroundColor Green
