from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from prepare_road_vehicle_dataset import prepare_dataset
from swanlab_utils import maybe_add_swanlab_callback


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on NVIDIA CUDA GPUs.")
    parser.add_argument("--model", default="yolov8s.pt", help="Try yolov8m.pt or yolov8l.pt on A100 for higher accuracy.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0", help="Use '0' for one A100 or '0,1' for two A100s.")
    parser.add_argument("--project", default="runs_a100")
    parser.add_argument("--name", default="road_vehicle_yolov8s_a100")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source-dir", type=Path, default=None, help="Already downloaded raw Kaggle dataset directory.")
    parser.add_argument("--no-swanlab", action="store_true", help="Disable SwanLab experiment logging.")
    args = parser.parse_args()

    data_yaml = prepare_dataset(args.data_dir, args.source_dir)

    model = YOLO(args.model)
    is_single_device = "," not in str(args.device)
    maybe_add_swanlab_callback(
        model,
        enabled=(not args.no_swanlab) and is_single_device,
        project="hw2-road-vehicle-yolov8",
        experiment_name=args.name,
        description="YOLOv8 fine-tuning on Road Vehicle Images Dataset using NVIDIA A100.",
    )
    if not is_single_device and not args.no_swanlab:
        print("SwanLab callback skipped for multi-GPU DDP. Use --device 0 for SwanLab logging.")

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=8,
        project=args.project,
        name=args.name,
        pretrained=True,
        patience=30,
        cache=True,
        amp=True,
    )


if __name__ == "__main__":
    main()
