"""PyInstaller onedir spec.

Build this file on the target OS.  PyInstaller does not produce a portable
cross-OS binary, so the same spec is intentionally used by Windows, macOS and
Linux jobs with their native Python/Tk installation.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


ROOT = Path(__file__).resolve().parent.parent
datas = [(str(ROOT / name), ".") for name in
         ("config.default.json", "models.default.json")
         if (ROOT / name).is_file()]
datas += collect_data_files("tkinterdnd2")
hiddenimports = collect_submodules("tkinterdnd2")

a = Analysis(
    [str(ROOT / "llama_manager.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="llama-manager",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="llama-manager",
)
