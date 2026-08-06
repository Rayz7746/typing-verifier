# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


mediapipe_datas, mediapipe_binaries, mediapipe_hiddenimports = collect_all(
    "mediapipe"
)

# A venv created from Anaconda still loads several extension-module DLLs from
# the base installation. PyInstaller does not recognize that hybrid layout as
# Conda, so include only the runtime DLLs its dependency scan otherwise misses.
conda_runtime_binaries = []
conda_bin_dir = Path(sys.base_prefix) / "Library" / "bin"
for dll_name in ("ffi.dll", "liblzma.dll", "libexpat.dll", "tcl86t.dll", "tk86t.dll"):
    dll_path = conda_bin_dir / dll_name
    if dll_path.is_file():
        conda_runtime_binaries.append((str(dll_path), "."))

# PyInstaller cannot query Qt's plugin paths in this Anaconda-derived venv.
# Collect the required Windows platform plugin directly at the location used
# by the standard PySide6 runtime hook.
pyside_package_dir = Path(sys.prefix) / "Lib" / "site-packages" / "PySide6"
qwindows_plugin = pyside_package_dir / "plugins" / "platforms" / "qwindows.dll"
if not qwindows_plugin.is_file():
    raise FileNotFoundError(f"Qt Windows platform plugin not found: {qwindows_plugin}")
pyside_plugin_binaries = [
    (str(qwindows_plugin), str(Path("PySide6") / "plugins" / "platforms"))
]

analysis = Analysis(
    ["ui_app.py"],
    pathex=[],
    binaries=mediapipe_binaries + conda_runtime_binaries + pyside_plugin_binaries,
    datas=mediapipe_datas
    + [("models/hand_landmarker.task", "models")],
    hiddenimports=mediapipe_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="TypingVerifier",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory="_internal",
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TypingVerifier",
)
