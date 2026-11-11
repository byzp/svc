"""Detect the optional CUDA compiler once, before loading training extensions."""
from functools import lru_cache
import os
from pathlib import Path
import subprocess
import warnings


@lru_cache(maxsize=1)
def cuda_compiler_available():
    # Match cpp_extension's CUDA_HOME selection: hiding nvcc from PATH alone
    # does not disable a compiler installed under /usr/local/cuda.
    from torch.utils.cpp_extension import CUDA_HOME

    compiler = Path(CUDA_HOME) / 'bin' / ('nvcc.exe' if os.name == 'nt' else 'nvcc') if CUDA_HOME else None
    available = compiler is not None and compiler.is_file() and os.access(compiler, os.X_OK)
    if available:
        try:
            subprocess.run([str(compiler), '--version'], check=True, timeout=10,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            available = False
    if not available:
        warnings.warn(
            'nvcc is unavailable; CUDA training operations will use native PyTorch. '
            'FP16 AMP and the training configuration are unchanged.',
            RuntimeWarning, stacklevel=2,
        )
    return available
