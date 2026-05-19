# Road Vehicle YOLOv8 Fine-tuning

This folder contains two training entrypoints for the Kaggle Road Vehicle Images Dataset:

- `scripts/train_m4_pro.py`: Apple Silicon M4 Pro / MPS training.
- `scripts/train_a100.py`: NVIDIA A100 80G / CUDA training.

Both scripts download the Kaggle dataset, convert or mirror it into the YOLOv8 folder layout, write `data/road_vehicle_yolo/road_vehicle.yaml`, download the selected YOLOv8 pretrained model automatically, and fine-tune it.

## 1. Environment

Create and activate a Python environment:

```bash
cd /Users/bytedance/Private_Czw/homework/hw2_yolov8_vehicle
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Enable SwanLab logging by setting the API key as an environment variable:

```bash
export SWANLAB_API_KEY="your_swanlab_api_key"
```

For Kaggle download, make sure your Kaggle credentials are available. Common options:

- Put `kaggle.json` in `~/.kaggle/kaggle.json`.
- Or log in through the Kaggle/kagglehub flow if your environment supports it.

## 2. M4 Pro Training

Smoke test first:

```bash
cd /Users/bytedance/Private_Czw/homework/hw2_yolov8_vehicle
source .venv/bin/activate
python scripts/train_m4_pro.py --model yolov8n.pt --epochs 5 --batch 8
```

Recommended course-project run:

```bash
python scripts/train_m4_pro.py --model yolov8s.pt --epochs 80 --batch 8
```

If memory pressure is high, reduce `--batch` to `4`. If training is too slow, use `yolov8n.pt`.

Expected output:

```text
runs_m4/road_vehicle_yolov8s_m4/weights/best.pt
```

If the current Python environment cannot access MPS, use the local fallback script:

```bash
python scripts/train_local.py --model yolov8n.pt --epochs 30 --imgsz 512 --batch 4
```

Expected output:

```text
runs_local/road_vehicle_yolov8n_local/weights/best.pt
```

## 3. A100 80G Training

Single A100, recommended:

```bash
cd /path/to/hw2_yolov8_vehicle
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/train_a100.py --model yolov8s.pt --epochs 100 --batch 64 --device 0
```

Use a single A100 if you need SwanLab curves for the report:

```bash
export SWANLAB_API_KEY="your_swanlab_api_key"
python scripts/train_a100.py --model yolov8s.pt --epochs 100 --batch 64 --device 0
```

Higher-accuracy run:

```bash
python scripts/train_a100.py --model yolov8m.pt --epochs 100 --batch 48 --device 0
```

Two A100s:

```bash
python scripts/train_a100.py --model yolov8m.pt --epochs 100 --batch 96 --device 0,1
```

For this 3004-image dataset, one A100 is usually enough. Two GPUs may not speed up much because the dataset is small.
The built-in SwanLab callback is enabled for single-device runs. For multi-GPU DDP, the script skips SwanLab unless you modify Ultralytics integration callbacks as described in the SwanLab documentation.

Expected output:

```text
runs_a100/road_vehicle_yolov8s_a100/weights/best.pt
```

## 3.1 Offline Transfer to A100

If Kaggle download is slow on the A100 machine, download and package assets on a faster local machine:

```bash
cd /Users/bytedance/Private_Czw/homework/hw2_yolov8_vehicle
source .venv/bin/activate
python scripts/download_and_bundle_assets.py --model yolov8s.pt
```

Copy `road_vehicle_assets_for_a100.tar.gz` to the A100 project folder, then unpack:

```bash
cd /dcar_ai_vepfs/zwchen/ccopen/hw2_yolov8_vehicle
tar -xzf road_vehicle_assets_for_a100.tar.gz
```

Train without Kaggle download:

```bash
python scripts/train_a100.py \
  --model yolov8s.pt \
  --epochs 100 \
  --batch 64 \
  --device 0 \
  --source-dir data/road_vehicle_yolo
```

## 4. Validate and Predict

Validate:

```bash
yolo detect val model=runs_a100/road_vehicle_yolov8s_a100/weights/best.pt data=data/road_vehicle_yolo/road_vehicle.yaml imgsz=640 device=0
```

Predict on an image or folder:

```bash
yolo detect predict model=runs_a100/road_vehicle_yolov8s_a100/weights/best.pt source=/path/to/images imgsz=640 conf=0.25 save=True
```

Track a video for the next homework step:

```bash
yolo track model=runs_a100/road_vehicle_yolov8s_a100/weights/best.pt source=/path/to/test_video.mp4 imgsz=640 conf=0.25 tracker=bytetrack.yaml save=True
```
