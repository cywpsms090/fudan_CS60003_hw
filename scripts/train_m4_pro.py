from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from prepare_road_vehicle_dataset import prepare_dataset
from swanlab_utils import maybe_add_swanlab_callback


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on Apple Silicon MPS.")
    parser.add_argument("--model", default="yolov8s.pt", help="Use yolov8n.pt for fastest smoke tests.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="runs_m4")
    parser.add_argument("--name", default="road_vehicle_yolov8s_m4")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source-dir", type=Path, default=None, help="Already downloaded raw Kaggle dataset directory.")
    parser.add_argument("--no-swanlab", action="store_true", help="Disable SwanLab experiment logging.")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available. Use the A100 script or install a PyTorch build with MPS support.")

    data_yaml = prepare_dataset(args.data_dir, args.source_dir)

    model = YOLO(args.model)
    maybe_add_swanlab_callback(
        model,
        enabled=not args.no_swanlab,
        project="hw2-road-vehicle-yolov8",
        experiment_name=args.name,
        description="YOLOv8 fine-tuning on Road Vehicle Images Dataset using Apple Silicon MPS.",
    )
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="mps",
        workers=4,
        project=args.project,
        name=args.name,
        pretrained=True,
        patience=20,
        cache=False,
    )


if __name__ == "__main__":
    main()
