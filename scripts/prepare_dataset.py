"""Reorganiza los datasets descargados en `data_raw/` a la estructura que
espera `ml.training.dataset.DeepfakeDataset`: `data/real/*.jpg` y
`data/ai_generated/*.jpg`.

Fuentes soportadas:
  - DeepFakeFace (Hugging Face, OpenRL): data_raw/deepfakeface/*.zip
    30k reales (wiki) + 90k generadas por difusión (text2img/inpainting/insight)
  - 140k Real and Fake Faces (Kaggle, xhlulu): data_raw/140k/
    70k reales + 70k generadas por StyleGAN (GAN)

Los nombres de archivo se prefijan con la fuente porque las cuatro carpetas
de DeepFakeFace comparten el mismo nombre base (misma persona, distinto
generador) — sin el prefijo se pisarían entre sí al copiar a una carpeta
plana.

Uso:
    python scripts/prepare_dataset.py --source deepfakeface
    python scripts/prepare_dataset.py --source kaggle140k
    python scripts/prepare_dataset.py --source all
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_OUT = PROJECT_ROOT / "data"

_DEEPFAKEFACE_SOURCES = {
    "wiki": ("real", "wiki"),
    "text2img": ("ai_generated", "text2img"),
    "inpainting": ("ai_generated", "inpainting"),
    "insight": ("ai_generated", "insight"),
}


def _copy_with_prefix(src_dir: Path, dst_dir: Path, prefix: str) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in src_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            shutil.copy2(path, dst_dir / f"{prefix}_{path.name}")
            count += 1
    return count


def prepare_deepfakeface(cleanup: bool = True) -> None:
    zip_root = DATA_RAW / "deepfakeface"
    extract_root = zip_root / "_extracted"

    for zip_name, (label, prefix) in _DEEPFAKEFACE_SOURCES.items():
        zip_path = zip_root / f"{zip_name}.zip"
        if not zip_path.exists():
            print(f"  [saltado] no existe {zip_path}")
            continue

        print(f"Extrayendo {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_root)

        src_dir = extract_root / zip_name
        dst_dir = DATA_OUT / label
        count = _copy_with_prefix(src_dir, dst_dir, prefix)
        print(f"  -> {count} imágenes copiadas a data/{label}/ (prefijo '{prefix}_')")

    if cleanup:
        print("Limpiando archivos intermedios (zips + extracción)...")
        shutil.rmtree(zip_root, ignore_errors=True)


def prepare_kaggle140k(cleanup: bool = True) -> None:
    """Espera la estructura que trae el dataset de Kaggle tras
    `kaggle datasets download ... --unzip`:
    `data_raw/140k/real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/*.jpg`
    """
    root = DATA_RAW / "140k" / "real_vs_fake" / "real-vs-fake"
    if not root.exists():
        print(f"  [saltado] no existe {root} (¿ya corriste el download de Kaggle?)")
        return

    for split_dir in root.iterdir():
        if not split_dir.is_dir():
            continue
        for class_name, label in (("real", "real"), ("fake", "ai_generated")):
            src_dir = split_dir / class_name
            if not src_dir.is_dir():
                continue
            dst_dir = DATA_OUT / label
            count = _copy_with_prefix(src_dir, dst_dir, f"stylegan_{split_dir.name}")
            print(f"  -> {count} imágenes copiadas a data/{label}/ (140k/{split_dir.name}/{class_name})")

    if cleanup:
        print("Limpiando archivos intermedios de Kaggle...")
        shutil.rmtree(DATA_RAW / "140k", ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["deepfakeface", "kaggle140k", "all"], default="all")
    parser.add_argument("--keep-raw", action="store_true", help="No borrar los archivos intermedios")
    args = parser.parse_args()
    cleanup = not args.keep_raw

    if args.source in ("deepfakeface", "all"):
        prepare_deepfakeface(cleanup=cleanup)
    if args.source in ("kaggle140k", "all"):
        prepare_kaggle140k(cleanup=cleanup)

    for label in ("real", "ai_generated"):
        n = len(list((DATA_OUT / label).glob("*"))) if (DATA_OUT / label).exists() else 0
        print(f"Total en data/{label}/: {n}")


if __name__ == "__main__":
    main()
