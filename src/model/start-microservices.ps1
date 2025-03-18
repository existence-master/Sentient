# Set the base directory to the script's location (src/model)
$BaseDir = $PSScriptRoot

# Path to the virtual environment activation script (use Activate.ps1 for PowerShell)
$VenvActivate = Join-Path -Path $BaseDir -ChildPath "venv\Scripts\Activate.ps1"

# Define services and their corresponding script names
$Services = @{
    "app"     = "app.py"
    "auth"    = "auth.py"
    "chat"    = "chat.py"
    "common"  = "common.py"
    "memory"  = "memory.py"
    "scraper" = "scraper.py"
    "utils"   = "utils.py"
}

# Debugging: Print the detected paths
Write-Host "Base Directory: $BaseDir"
Write-Host "Virtual Environment Activate Script: $VenvActivate"
Write-Host "Checking for scripts in service directories..."

# Check if the venv activate script exists
if (-not (Test-Path $VenvActivate)) {
    Write-Error "Virtual environment activate script not found at $VenvActivate"
    exit 1
}

# Start each service in a new terminal with a custom window title
foreach ($Service in $Services.Keys) {
    $ServiceDir = Join-Path -Path $BaseDir -ChildPath $Service
    $ScriptName = $Services[$Service]
    $ScriptPath = Join-Path -Path $ServiceDir -ChildPath $ScriptName

    # Debugging: Print the script path
    Write-Host "Looking for: $ScriptPath"

    # Check if the script exists before launching
    if (Test-Path $ScriptPath) {
        Write-Host "Starting ${Service}..."

        # Set the window title to the service name (e.g., "app", "auth")
        $windowTitle = $Service

        # Define the Python debug command separately for clarity
        $pythonDebugCommand = 'import sys; print("Using Python: {}".format(sys.executable))'

        # Build the command to run in the new PowerShell window
        $command = "& { `$host.UI.RawUI.WindowTitle = '$windowTitle'; . `"$VenvActivate`"; cd `"$ServiceDir`"; python `"$ScriptName`" }"

        # Start a new PowerShell window and keep it open
        Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $command
    } else {
        Write-Host "Skipping ${Service}: Script not found at $ScriptPath"
    }
}

Write-Host "All services are starting..."