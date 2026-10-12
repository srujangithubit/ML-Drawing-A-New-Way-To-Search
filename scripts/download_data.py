from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

sys.path.insert(0, str(ROOT))
from src.data import class_filename, read_classes  # noqa: E402


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "drawing-search-course-project/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                downloaded += len(block)
                if total:
                    percent = downloaded * 100 / total
                    print(f"\r  {percent:5.1f}% ({downloaded / (1024**2):.1f} MiB)", end="", flush=True)
            if total:
                print()
    except Exception:
        if destination.exists():
            destination.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Download QuickDraw NumPy bitmap files.")
    parser.add_argument("--classes-file", default=str(ROOT / "config" / "classes.txt"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "numpy_bitmap"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    classes = read_classes(args.classes_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in classes:
        destination = output_dir / class_filename(label)
        if destination.exists() and not args.overwrite:
            print(f"[skip] {label}: {destination} already exists")
            continue
        url = BASE_URL + quote(f"{label}.npy", safe="")
        print(f"[download] {label}\n  {url}")
        try:
            download(url, destination)
            print(f"[saved] {destination} ({destination.stat().st_size / (1024**2):.1f} MiB)")
        except urllib.error.HTTPError as exc:
            print(f"[error] HTTP {exc.code} for '{label}'. Verify the category spelling in the class list.", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"[error] Could not download '{label}': {exc}", file=sys.stderr)
            return 1
    print("Data download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
