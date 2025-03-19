#use Set-ExecutionPolicy RemoteSigned -Scope CurrentUser to allow running scripts
$excludeFolders = @("venv", "__pycache__", "node_modules", ".next")
$path = "D:\Documents\cyber\projects\Sentient-New\Code\src\model"

function Get-Tree {
    param ($directory, $level = 0)

    $items = Get-ChildItem $directory
    foreach ($item in $items) {
        # Skip excluded folders
        if ($excludeFolders -contains $item.Name) {
            continue
        }

        # Output the item with indentation based on its level
        Write-Host (" " * $level + $item.Name)

        # Recursively get sub-items
        if ($item.PSIsContainer) {
            Get-Tree $item.FullName ($level + 1)
        }
    }
}

Get-Tree $path
