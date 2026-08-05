"""Prepare the Windows DLL search state before importing PySide6."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


_SYSTEM_ICU = None


def prepare_qt_runtime() -> None:
    """Prevent Anaconda's versioned ICU shim from shadowing Windows ICU."""

    global _SYSTEM_ICU
    if sys.platform != "win32" or _SYSTEM_ICU is not None:
        return
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    system_icu = system_root / "System32" / "icuuc.dll"
    if system_icu.is_file():
        _SYSTEM_ICU = ctypes.WinDLL(str(system_icu))
