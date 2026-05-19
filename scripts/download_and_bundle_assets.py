from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from ultralytics import YOLO

from prepare_road_vehicle_dataset import prepare_dataset


def add_if_exists(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    if path.exists():
        tar.add(path, arcname=arcname)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset/model locally and create a bundle for offline A100 training.")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--bundle", type=Path, default=Path("road_vehicle_assets_for_a100.tar.gz"))
    args = parser.parse_args()

    data_yaml = prepare_dataset(args.data_dir)

    # Loading YOLO downloads the pretrained .pt file into the current directory/cache if needed.
    model = YOLO(args.model)
    model_path = Path(args.model)
    if not model_path.exists() and hasattr(model, "ckpt_path"):
        model_path = Path(model.ckpt_path)

    with tarfile.open(args.bundle, "w:gz") as tar:
        tar.add(args.data_dir / "road_vehicle_yolo", arcname="data/road_vehicle_yolo")
        add_if_exists(tar, model_path, Path(model_path).name)

    print(f"Data config: {data_yaml.resolve()}")
    print(f"Model file: {model_path.resolve() if model_path.exists() else args.model}")
    print(f"Bundle written: {args.bundle.resolve()}")


if __name__ == "__main__":
    main()
