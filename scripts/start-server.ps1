param([switch]$CheckOnly, [switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$environmentCheck = Join-Path $PSScriptRoot 'environment_check.py'
$url = 'http://127.0.0.1:8765/'
$alreadyRunning = $false
try {
    $health = Invoke-RestMethod -Uri ($url + 'api/health') -TimeoutSec 3
    $page = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
    $alreadyRunning = ($health.ok -eq $true -and $page.Content -match '<title>TeraSort')
} catch {
    # No responding dashboard: the server below reports any port conflict.
}
if ($alreadyRunning -and -not $CheckOnly) {
    Write-Host "TeraSort is already running: $url"
    if (-not $NoBrowser -and -not $CheckOnly) { Start-Process $url }
    exit 0
}

if ($CheckOnly -and -not (Test-Path -LiteralPath $python)) {
    throw 'Environment not found. Run install_and_start.bat to install it.'
}

if (-not (Test-Path -LiteralPath $python)) {
    if ($CheckOnly) { throw 'Environment not found.' }
    Write-Host 'Environment is missing. Installing TeraSort and its dependencies...'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'install.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Automatic environment installation failed.' }
}

& $python $environmentCheck packages
$environmentReady = ($LASTEXITCODE -eq 0)
if (-not $environmentReady) {
    if ($CheckOnly) {
        throw 'One or more required packages are missing or incompatible. Run install_and_start.bat to repair the environment.'
    }
    Write-Host 'Required packages are missing or incompatible. Repairing the environment...'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'install.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Automatic dependency repair failed.' }
    & $python $environmentCheck packages
    if ($LASTEXITCODE -ne 0) { throw 'Required dependencies are still unavailable after repair.' }
}

& $python $environmentCheck progress-smoke
if ($LASTEXITCODE -ne 0) { throw 'Three-phase progress hooks are unavailable. Run install_and_start.bat to repair the environment.' }

if ($CheckOnly) {
    Write-Host "Environment ready. Start the dashboard with start_server.bat ($url)."
    exit 0
}
Write-Host "Starting TeraSort: $url"
Write-Host 'Keep this window open while using the server. Press Ctrl+C to stop it.'
Push-Location $root
try {
    if ($NoBrowser) { & $python -u -m terasort.cli web --no-browser }
    else { & $python -u -m terasort.cli web }
    $serverExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $serverExit
