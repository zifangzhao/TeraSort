# Environment repair and GPU validation, 2026-09-24

The virtual environment referenced a missing Python interpreter beneath
`C:/Frank/Code/PainProject/.runtime-python`. It now uses a runtime inside
TeraSort:

`C:/Frank/Code/TeraSort/.runtime-python/cpython-3.11.16/python/python.exe`

The Python 3.11.16 standalone archive came from the
[Astral 20260901 release](https://github.com/astral-sh/python-build-standalone/releases/tag/20260901).
Its SHA-256 was checked against the release asset digest:

`06cbe479e039f5b9cb5640c286d790074d63f549f92a32d599a3748293bd4510`

`python -m venv --upgrade .venv` repaired the interpreter paths while retaining
the installed packages. The original configuration is backed up at
`.venv/pyvenv.cfg.before-runtime-repair`. The runtime directory is ignored by
Git. The installer now recognizes this project-local runtime.

Verified installed stack:

- Python 3.11.16, 64-bit
- PyTorch 2.10.0+cu128
- CuPy 14.2.0
- Kilosort 4.1.7
- NVIDIA GeForce RTX 5060 Ti, 16 GB
- `pip check`: no broken requirements
- `terasort backends`: standard, deep_tiled, cublas

The CUDA detector executed successfully both with and without importing
PyTorch. CuPy emitted a toolkit-location warning in the standalone process,
but NVRTC kernel compilation and execution succeeded using installed runtime
packages. The Visual Studio 2019 C++ toolchain was found with `vswhere`; a new
compiler installation was unnecessary.

The initial GPU process stalled in the restricted execution sandbox and was
terminated. The same smoke test succeeded in the normal GPU execution
environment. The first full pytest attempt hit permissions on an old shared
temporary directory. With a new project-local test directory, all 158 tests
passed. An additional GPU crash/resume test then passed, comparing candidate,
spike, and template arrays exactly. Use `scripts/test.ps1` to select a fresh
temporary directory automatically.

Final regression after the calibration-memory fix and GPU resume test:
**159 passed in 8.54 seconds**, using `scripts/test.ps1`.
