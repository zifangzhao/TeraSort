param([string]$Python)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$venv = Join-Path $root '.venv'
if ($Python) {
    $basePython = (Resolve-Path -LiteralPath $Python -ErrorAction Stop).Path
    $baseArgs = @()
} else {
    $basePython = 'py'
    $baseArgs = @('-3.11')
}
try {
    & $basePython @baseArgs -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,11) and sys.maxsize > 2**32 else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'Wrong Python version or architecture' }
} catch {
    throw "A 64-bit Python 3.11 executable is required. Install it or pass -Python 'C:\path\to\python.exe'."
}
& $basePython @baseArgs -m venv $venv
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
$venvPython = Join-Path $venv 'Scripts/python.exe'
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip update failed' }
& $venvPython -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
& $venvPython -m pip install -e $root
if ($LASTEXITCODE -ne 0) { throw 'TeraSort installation failed' }
& $venvPython -c 'import cupy as cp,sys,torch,terasort; from terasort.candidates.detectors import CudaDetector; q=cp.zeros((16,2),cp.float32); q[3,0]=5; found=cp.asnumpy(CudaDetector().detect(q,floor=3)); print(terasort.__version__,torch.cuda.is_available(),found.tolist()); sys.exit(0 if torch.cuda.is_available() and found.tolist()==[6] else 1)'
if ($LASTEXITCODE -ne 0) { throw 'CUDA kernel smoke test failed' }
& (Join-Path $venv 'Scripts/terasort.exe') backends
if ($LASTEXITCODE -ne 0) { throw 'Backend check failed' }
