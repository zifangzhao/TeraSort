param([string]$Python)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$venv = Join-Path $root '.venv'
$venvPython = Join-Path $venv 'Scripts/python.exe'
$pythonProbe = 'import json,struct,sys; print(json.dumps({"major":sys.version_info.major,"minor":sys.version_info.minor,"bits":struct.calcsize("P")*8}))'

function Get-PythonInfo([string]$Executable, [string[]]$Arguments = @()) {
    try {
        $output = & $Executable @Arguments -c $pythonProbe 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        return ($output -join "`n" | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function Test-SupportedPython($Info) {
    return ($null -ne $Info -and $Info.major -eq 3 -and
            $Info.minor -ge 10 -and $Info.minor -le 14 -and $Info.bits -eq 64)
}

$basePython = $null
$baseArgs = @()
$baseInfo = $null
if ($Python) {
    $basePython = (Resolve-Path -LiteralPath $Python -ErrorAction Stop).Path
    $baseInfo = Get-PythonInfo $basePython
    if (-not (Test-SupportedPython $baseInfo)) {
        throw 'TeraSort supports 64-bit Python 3.10 through 3.14. Choose a compatible interpreter.'
    }
}

if (Test-Path -LiteralPath $venvPython) {
    $existingInfo = Get-PythonInfo $venvPython
    if (-not (Test-SupportedPython $existingInfo)) {
        throw "The existing .venv is not a supported 64-bit Python 3.10–3.14 environment. Move or remove '$venv' before reinstalling."
    }
    if ($baseInfo -and ($baseInfo.major -ne $existingInfo.major -or $baseInfo.minor -ne $existingInfo.minor)) {
        throw "The existing .venv uses Python $($existingInfo.major).$($existingInfo.minor); it cannot be switched to $($baseInfo.major).$($baseInfo.minor) in place. Move or remove '$venv' first."
    }
    Write-Host "Using existing 64-bit Python $($existingInfo.major).$($existingInfo.minor) environment."
} else {
    if (Test-Path -LiteralPath $venv) {
        throw "The existing environment folder '$venv' is incomplete. Move or remove it before reinstalling."
    }
    if (-not $basePython) {
        $localRuntime = Join-Path $root '.runtime-python/cpython-3.11.16/python/python.exe'
        if (Test-Path -LiteralPath $localRuntime) {
            $localInfo = Get-PythonInfo $localRuntime
            if (Test-SupportedPython $localInfo) {
                $basePython, $baseInfo = $localRuntime, $localInfo
            }
        }
    }
    if (-not $basePython -and (Get-Command py.exe -ErrorAction SilentlyContinue)) {
        foreach ($minor in @(14, 13, 12, 11, 10)) {
            $candidateArgs = @("-3.$minor")
            $candidateInfo = Get-PythonInfo 'py' $candidateArgs
            if (Test-SupportedPython $candidateInfo) {
                $basePython, $baseArgs, $baseInfo = 'py', $candidateArgs, $candidateInfo
                break
            }
        }
    }
    if (-not $basePython) {
        $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($pythonCommand) {
            $candidateInfo = Get-PythonInfo $pythonCommand.Source
            if (Test-SupportedPython $candidateInfo) {
                $basePython, $baseInfo = $pythonCommand.Source, $candidateInfo
            }
        }
    }
    if (-not $basePython) {
        throw "A 64-bit Python 3.10–3.14 executable is required. Install one or pass -Python 'C:\path\to\python.exe'."
    }
    Write-Host "Creating environment with Python $($baseInfo.major).$($baseInfo.minor)."
    & $basePython @baseArgs -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}

& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw 'pip tooling update failed' }

$torchInfo = $null
try {
    $torchOutput = & $venvPython -c 'import json,torch; print(json.dumps({"version":torch.__version__.split("+",1)[0],"cuda":torch.version.cuda}))' 2>$null
    if ($LASTEXITCODE -eq 0) { $torchInfo = $torchOutput -join "`n" | ConvertFrom-Json }
} catch { }
if (-not $torchInfo -or $torchInfo.version -ne '2.10.0' -or $torchInfo.cuda -ne '12.8') {
    Write-Host 'Installing the supported CUDA 12.8 PyTorch build...'
    & $venvPython -m pip install --force-reinstall torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
} else {
    Write-Host 'Compatible CUDA PyTorch is already installed.'
}

Write-Host 'Installing or repairing TeraSort and its required packages...'
& $venvPython -m pip install -e $root
if ($LASTEXITCODE -ne 0) { throw 'TeraSort or one of its required packages could not be installed' }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Python dependencies remain incompatible after installation' }
& $venvPython -c 'import cupy as cp,sys,torch,terasort; from terasort.candidates.detectors import CudaDetector; q=cp.zeros((16,2),cp.float32); q[3,0]=5; found=cp.asnumpy(CudaDetector().detect(q,floor=3)); print(terasort.__version__,torch.cuda.is_available(),found.tolist()); sys.exit(0 if torch.cuda.is_available() and found.tolist()==[6] else 1)'
if ($LASTEXITCODE -ne 0) { throw 'CUDA dependency or kernel check failed' }
& (Join-Path $venv 'Scripts/terasort.exe') backends
if ($LASTEXITCODE -ne 0) { throw 'Backend check failed' }
