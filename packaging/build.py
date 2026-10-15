"""Build and optionally zip the native onedir bundle."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", action="store_true", help="create a ZIP beside dist/")
    args = parser.parse_args()

    command = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        str(ROOT / "packaging" / "llama-manager.spec"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    if args.zip:
        source = DIST / "llama-manager"
        if not source.is_dir():
            raise SystemExit(f"PyInstaller output missing: {source}")
        archive = shutil.make_archive(
            str(DIST / "llama-manager"), "zip", root_dir=source.parent,
            base_dir=source.name)
        print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
