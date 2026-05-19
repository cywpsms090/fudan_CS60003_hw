from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from prepare_road_vehicle_dataset import prepare_dataset
from swanlab_utils import maybe_add_swanlab_callback


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 locally using MPS when available, otherwise CPU.")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--project", default="runs_local")
    parser.add_argument("--name", default="road_vehicle_yolov8n_local")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--no-swanlab", action="store_true")
    args = parser.parse_args()

    data_yaml = prepare_dataset(args.data_dir, args.source_dir)
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this Python environment.")

    model = YOLO(args.model)
    maybe_add_swanlab_callback(
        model,
        enabled=not args.no_swanlab,
        project="hw2-road-vehicle-yolov8",
        experiment_name=args.name,
        description=f"YOLOv8 local fine-tuning on Road Vehicle Images Dataset using {device}.",
    )
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=2,
        project=args.project,
        name=args.name,
        pretrained=True,
        patience=10,
        cache=False,
    )


if __name__ == "__main__":
    main()
