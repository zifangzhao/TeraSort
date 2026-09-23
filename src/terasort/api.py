"""A drop-in Kilosort API with scoped, source-guarded CUDA hot paths."""

from contextlib import ExitStack
from pathlib import Path
import sys

import numpy as np


def available_backends():
    """Report selectable backends without creating a CUDA context."""
    native = (sys.platform == "win32" and
              (Path(__file__).with_name("_native") / "cublas_bridge.dll").is_file())
    if native:
        try:
            import torch
        except ImportError:
            native = False
        else:
            native = ((torch.version.cuda or "").startswith("12.") and
                      (Path(torch.__file__).parent / "lib/cublas64_12.dll").is_file())
    return ("standard", "deep_tiled", "cublas") if native else ("standard", "deep_tiled")


def _select_backend(backend, device, settings):
    if backend not in ("auto", "standard", "deep_tiled", "cublas"):
        raise ValueError(f"Unknown backend {backend!r}")
    if backend == "standard":
        return backend
    specialized_geometry = (settings.get("n_templates", 6) == 6 and
                            settings.get("nt", 61) == 61)
    import torch
    requested_device = torch.device(device) if device is not None else None
    cuda = torch.cuda.is_available() and (requested_device is None or requested_device.type == "cuda")
    if backend == "auto":
        return "cublas" if cuda and specialized_geometry and "cublas" in available_backends() else "standard"
    if not cuda:
        raise RuntimeError(f"{backend} requires a CUDA-capable PyTorch installation and GPU")
    if backend not in available_backends():
        raise RuntimeError("The Windows x64 cuBLAS bridge is unavailable on this platform")
    if backend == "cublas" and not specialized_geometry:
        raise ValueError("The cuBLAS path requires n_templates=6 and nt=61")
    return backend


def _use_int16_reader(filename, file_object, data_dtype, backend, fast_int16):
    if not fast_int16 or backend == "standard" or file_object is not None:
        return False
    if filename is None or isinstance(filename, (list, tuple)):
        return False
    return np.dtype("int16" if data_dtype is None else data_dtype) == np.dtype("int16")


def run_kilosort(settings, probe=None, probe_name=None, filename=None,
                 data_dir=None, file_object=None, results_dir=None,
                 data_dtype=None, do_CAR=True, invert_sign=False, device=None,
                 progress_bar=None, save_extra_vars=False, clear_cache=False,
                 save_preprocessed_copy=False, bad_channels=None, shank_idx=None,
                 verbose_console=False, verbose_log=False, torch_thread_lim=None,
                 *, backend="auto", fast_int16=True,
                 skip_drift_correction=False):
    """Run Kilosort with the same input arguments, return tuple and Phy files.

    ``backend='auto'`` selects the tested Windows x64 cuBLAS path when CUDA is
    available and otherwise uses unmodified Kilosort. ``backend='standard'``
    always calls unmodified Kilosort. ``deep_tiled`` uses the portable CuPy CUDA
    kernels, while ``cublas`` requires the included Windows x64 native bridge.
    The INT16 reader applies only to a single named read-only INT16 binary.
    ``skip_drift_correction=True`` sets Kilosort's ``nblocks=0`` for this run
    without changing the caller's settings dictionary. This skips drift
    estimation and its extra detection pass; use it only when appropriate for
    the recording's motion.
    """
    import kilosort

    run_settings = ({**(settings or {}), "nblocks": 0}
                    if skip_drift_correction else settings)
    selected = _select_backend(backend, device, run_settings or {})
    with ExitStack() as stack:
        if selected == "cublas":
            from .kilosort_cublas import cublas_kilosort
            stack.enter_context(cublas_kilosort(device=device))
        elif selected == "deep_tiled":
            from .kilosort_tiled import tiled_kilosort
            stack.enter_context(tiled_kilosort(tile_samples=2048, deep=True))
        if _use_int16_reader(filename, file_object, data_dtype, selected, fast_int16):
            from .kilosort_int16 import native_int16_reader
            stack.enter_context(native_int16_reader(filename))
        return kilosort.run_kilosort(
            run_settings, probe=probe, probe_name=probe_name, filename=filename,
            data_dir=data_dir, file_object=file_object, results_dir=results_dir,
            data_dtype=data_dtype, do_CAR=do_CAR, invert_sign=invert_sign,
            device=device, progress_bar=progress_bar,
            save_extra_vars=save_extra_vars, clear_cache=clear_cache,
            save_preprocessed_copy=save_preprocessed_copy,
            bad_channels=bad_channels, shank_idx=shank_idx,
            verbose_console=verbose_console, verbose_log=verbose_log,
            torch_thread_lim=torch_thread_lim)
