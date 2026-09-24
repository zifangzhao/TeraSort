param([switch]$CheckOnly, [switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$url = 'http://127.0.0.1:8765/'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Environment not found. Run install_and_start.bat first.'
}
& $python -c 'import terasort, terasort.web'
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed. Run install_and_start.bat to repair it.' }

$alreadyRunning = $false
try {
    $health = Invoke-RestMethod -Uri ($url + 'api/health') -TimeoutSec 3
    $page = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
    $alreadyRunning = ($health.ok -eq $true -and $page.Content -match '<title>TeraSort')
} catch {
    # No responding dashboard: the server below reports any port conflict.
}
if ($alreadyRunning) {
    Write-Host "TeraSort is already running: $url"
    if (-not $NoBrowser -and -not $CheckOnly) { Start-Process $url }
    exit 0
}
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
