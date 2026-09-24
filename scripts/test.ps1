param([string[]]$PytestArgs = @())

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Install the project environment with scripts/install.ps1 first.'
}
# Use a new directory within the project; an old shared pytest directory can
# have an incompatible Windows ACL. Never clear an existing directory here.
$testRoot = Join-Path $root ('build/pytest-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $testRoot) { throw 'Test directory must be new' }
Push-Location $root
try {
    & $python -m pytest -q tests --tb=short --basetemp $testRoot @PytestArgs
    $testExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $testExit
