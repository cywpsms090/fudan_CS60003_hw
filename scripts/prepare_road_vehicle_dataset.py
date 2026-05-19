from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub
import yaml


DATASET_SLUG = "ashfakyeafi/road-vehicle-images-dataset"

CLASS_NAMES = [
    "car",
    "bus",
    "motorbike",
    "three wheelers -CNG-",
    "rickshaw",
    "truck",
    "pickup",
    "minivan",
    "suv",
    "van",
    "bicycle",
    "auto rickshaw",
    "human hauler",
    "wheelbarrow",
    "ambulance",
    "minibus",
    "taxi",
    "army vehicle",
    "scooter",
    "policecar",
    "garbagevan",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_split_dir(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        direct = root / name
        if direct.exists():
            return direct

    for candidate in root.rglob("*"):
        if candidate.is_dir() and candidate.name.lower() in names:
            return candidate

    raise FileNotFoundError(f"Could not find any split directory named {names} under {root}")


def _find_images_dir(split_dir: Path) -> Path:
    direct = split_dir / "images"
    if direct.exists():
        return direct
    if any(p.suffix.lower() in IMAGE_SUFFIXES for p in split_dir.iterdir() if p.is_file()):
        return split_dir
    for candidate in split_dir.rglob("*"):
        if candidate.is_dir() and candidate.name.lower() == "images":
            return candidate
    raise FileNotFoundError(f"Could not find images directory under {split_dir}")


def _find_labels_dir(split_dir: Path) -> Path:
    direct = split_dir / "labels"
    if direct.exists():
        return direct
    if any(p.suffix.lower() == ".txt" for p in split_dir.iterdir() if p.is_file()):
        return split_dir
    for candidate in split_dir.rglob("*"):
        if candidate.is_dir() and candidate.name.lower() == "labels":
            return candidate
    raise FileNotFoundError(f"Could not find labels directory under {split_dir}")


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def prepare_dataset(output_dir: Path, source_dir: Path | None = None) -> Path:
    kaggle_path = source_dir.resolve() if source_dir else Path(kagglehub.dataset_download(DATASET_SLUG))
    if not kaggle_path.exists():
        raise FileNotFoundError(f"Dataset source directory does not exist: {kaggle_path}")

    prepared_yaml = kaggle_path / "road_vehicle.yaml"
    if prepared_yaml.exists() and (kaggle_path / "images" / "train").exists() and (kaggle_path / "labels" / "train").exists():
        print(f"Using prepared YOLO dataset: {kaggle_path}")
        print(f"Data config: {prepared_yaml}")
        return prepared_yaml

    dataset_dir = output_dir / "road_vehicle_yolo"

    train_split = _find_split_dir(kaggle_path, ("train",))
    valid_split = _find_split_dir(kaggle_path, ("valid", "val", "validation"))

    train_images = _find_images_dir(train_split)
    train_labels = _find_labels_dir(train_split)
    valid_images = _find_images_dir(valid_split)
    valid_labels = _find_labels_dir(valid_split)

    _copy_tree(train_images, dataset_dir / "images" / "train")
    _copy_tree(train_labels, dataset_dir / "labels" / "train")
    _copy_tree(valid_images, dataset_dir / "images" / "val")
    _copy_tree(valid_labels, dataset_dir / "labels" / "val")

    data_yaml = dataset_dir / "road_vehicle.yaml"
    data = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"Kaggle cache path: {kaggle_path}")
    print(f"Prepared YOLO dataset: {dataset_dir.resolve()}")
    print(f"Data config: {data_yaml.resolve()}")
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Use an already downloaded Kaggle dataset directory instead of downloading.",
    )
    args = parser.parse_args()
    prepare_dataset(args.output_dir, args.source_dir)


if __name__ == "__main__":
    main()
