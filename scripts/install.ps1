$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venv = Join-Path $root '.venv'
py -3.11 -m venv $venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required' }
$python = Join-Path $venv 'Scripts/python.exe'
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip update failed' }
& $python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
& $python -m pip install -e $root
if ($LASTEXITCODE -ne 0) { throw 'TeraSort installation failed' }
& $python -c 'import torch, terasort; print("CUDA available:", torch.cuda.is_available()); print("TeraSort:", terasort.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'Installation check failed' }
