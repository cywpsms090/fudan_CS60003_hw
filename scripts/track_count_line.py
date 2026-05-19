from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


def side_of_line(point: tuple[float, float], line: tuple[int, int, int, int]) -> float:
    x, y = point
    x1, y1, x2, y2 = line
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def point_in_regions(point: tuple[float, float], regions: list[tuple[int, int, int, int]]) -> bool:
    x, y = point
    for x1, y1, x2, y2 in regions:
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if left <= x <= right and top <= y <= bottom:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 tracking plus virtual-line vehicle counting.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="outputs/tracked_counted.mp4")
    parser.add_argument("--csv", default="outputs/tracks.csv")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--line", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), required=True)
    parser.add_argument(
        "--ignore-region",
        nargs=4,
        type=int,
        action="append",
        default=[],
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Ignore tracked boxes whose center falls inside this fixed region. Can be repeated.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    model = YOLO(args.model)
    line = tuple(args.line)
    ignore_regions = [tuple(region) for region in args.ignore_region]
    previous_side: dict[int, float] = {}
    counted_ids: set[int] = set()
    trajectories: dict[int, list[tuple[int, int]]] = defaultdict(list)
    count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["frame", "track_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "cx", "cy", "crossed"])

        results = model.track(
            source=str(source),
            tracker=args.tracker,
            conf=args.conf,
            imgsz=args.imgsz,
            stream=True,
            persist=True,
            verbose=False,
        )

        for frame_idx, result in enumerate(results):
            frame = result.orig_img.copy()
            cv2.line(frame, line[:2], line[2:], (0, 255, 255), 3)

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.cpu().numpy().astype(int)
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                for box, track_id, class_id, conf in zip(xyxy, track_ids, class_ids, confs):
                    x1, y1, x2, y2 = box
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)
                    if point_in_regions((cx, cy), ignore_regions):
                        continue

                    current_side = side_of_line((cx, cy), line)
                    crossed = False

                    if track_id in previous_side and track_id not in counted_ids:
                        if previous_side[track_id] * current_side < 0:
                            counted_ids.add(track_id)
                            count += 1
                            crossed = True
                    previous_side[track_id] = current_side

                    class_name = model.names.get(class_id, str(class_id))
                    color = (0, 180, 0) if not crossed else (0, 0, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
                    cv2.putText(
                        frame,
                        f"{class_name} ID:{track_id}",
                        (int(x1), max(20, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                    trajectories[track_id].append((int(cx), int(cy)))
                    for p1, p2 in zip(trajectories[track_id][-20:-1], trajectories[track_id][-19:]):
                        cv2.line(frame, p1, p2, (255, 180, 0), 2)

                    csv_writer.writerow([
                        frame_idx,
                        track_id,
                        class_id,
                        class_name,
                        float(conf),
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                        cx,
                        cy,
                        int(crossed),
                    ])

            cv2.putText(frame, f"Crossing count: {count}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            writer.write(frame)

    writer.release()
    print(f"Output video: {output.resolve()}")
    print(f"Track CSV: {csv_path.resolve()}")
    print(f"Crossing count: {count}")


if __name__ == "__main__":
    main()
