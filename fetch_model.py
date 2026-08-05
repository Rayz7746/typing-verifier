"""Download the official MediaPipe Hand Landmarker model when needed."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SIZE = 7_819_105
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(path: Path = DEFAULT_MODEL_PATH, *, force: bool = False) -> Path:
    path = path.resolve()
    if (
        not force
        and path.is_file()
        and path.stat().st_size == MODEL_SIZE
        and _sha256(path) == MODEL_SHA256
    ):
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "typing-verifier/phase2"})

    print(f"Downloading official Hand Landmarker model to {path} ...")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        if temporary.stat().st_size != MODEL_SIZE:
            raise RuntimeError(
                f"model size mismatch: expected {MODEL_SIZE}, got {temporary.stat().st_size}"
            )
        actual_hash = _sha256(temporary)
        if actual_hash != MODEL_SHA256:
            raise RuntimeError(
                f"model SHA-256 mismatch: expected {MODEL_SHA256}, got {actual_hash}"
            )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    print(f"Model ready ({MODEL_SIZE:,} bytes).")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(ensure_model(args.output, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
