"""Giai đoạn DEV — trích subset ảnh từ zip NIH ChestX-ray14.

Vì zip không cho tải rời từng file, ta tải cả part zip rồi trích N ảnh đầu
ra thư mục bronze. Script này thuần Python (stdlib), không cần Spark.

Usage:
    python scripts/dev/extract_subset.py --zip data/raw/images/images_001.zip \
        --out data/bronze/images --limit 2000
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def extract_subset(zip_path: Path, out_dir: Path, limit: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        # Bỏ qua entry thư mục (kết thúc '/'), file AppleDouble '._*' và __MACOSX
        names = [
            n
            for n in zf.namelist()
            if not n.endswith("/")
            and n.lower().endswith(".png")
            and not Path(n).name.startswith("._")
            and "__MACOSX" not in n
        ]
        for name in names:
            if extracted >= limit:
                break
            target = out_dir / Path(name).name
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted += 1
    return extracted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    n = extract_subset(args.zip, args.out, args.limit)
    print(f"Extracted {n} images -> {args.out}")


if __name__ == "__main__":
    main()
