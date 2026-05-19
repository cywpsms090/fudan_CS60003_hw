#!/usr/bin/env python3
# Script purpose:
# Build a three-layer neural network classifier from scratch with NumPy
# to perform land-cover image classification on the EuroSAT_RGB dataset.

import argparse
import csv
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


# Configuration module:
# Store experiment hyperparameters in a single dataclass for easier reuse.
@dataclass
class Config:
    data_dir: Path
    output_dir: Path
    image_size: int = 16
    hidden_dim1: int = 256
    hidden_dim2: int = 128
    learning_rate: float = 0.03
    batch_size: int = 256
    epochs: int = 20
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    max_per_class: int = 0


# Utility module:
# Provide common helpers for reproducibility, filesystem setup and label handling.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    result = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    result[np.arange(labels.shape[0]), labels] = 1.0
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def classification_report_from_confusion(
    matrix: np.ndarray, class_names: List[str]
) -> Tuple[List[Dict[str, float]], float, float, float]:
    rows: List[Dict[str, float]] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1_scores: List[float] = []

    for idx, class_name in enumerate(class_names):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        support = float(matrix[idx, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        rows.append(
            {
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return rows, float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1_scores))


# Data processing module:
# Read images by class, resize and flatten them, then perform stratified splitting.
def load_dataset(config: Config) -> Tuple[np.ndarray, np.ndarray, List[str], List[Path]]:
    class_names = sorted(
        [entry.name for entry in config.data_dir.iterdir() if entry.is_dir()]
    )
    features: List[np.ndarray] = []
    labels: List[int] = []
    paths: List[Path] = []

    for label, class_name in enumerate(class_names):
        class_dir = config.data_dir / class_name
        image_paths = sorted(class_dir.glob("*.jpg"))
        if config.max_per_class > 0:
            image_paths = image_paths[: config.max_per_class]

        for image_path in image_paths:
            with Image.open(image_path) as image:
                resized = image.convert("RGB").resize(
                    (config.image_size, config.image_size), Image.Resampling.BILINEAR
                )
                array = np.asarray(resized, dtype=np.float32) / 255.0
                features.append(array.reshape(-1))
                labels.append(label)
                paths.append(image_path)

    x = np.stack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return x, y, class_names, paths


def stratified_split(
    y: np.ndarray, train_ratio: float, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for class_id in np.unique(y):
        indices = np.where(y == class_id)[0]
        rng.shuffle(indices)

        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train : n_train + n_val].tolist())
        test_indices.extend(indices[n_train + n_val : n_train + n_val + n_test].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return (
        np.array(train_indices, dtype=np.int64),
        np.array(val_indices, dtype=np.int64),
        np.array(test_indices, dtype=np.int64),
    )


def normalize_by_train(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        (x_train - mean) / std,
        (x_val - mean) / std,
        (x_test - mean) / std,
        mean,
        std,
    )


# Model module:
# Implement a three-layer fully connected neural network and manual backpropagation.
class ThreeLayerMLP:
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, output_dim: int):
        self.params = {
            "W1": np.random.randn(input_dim, hidden_dim1).astype(np.float32)
            * math.sqrt(2.0 / input_dim),
            "b1": np.zeros((1, hidden_dim1), dtype=np.float32),
            "W2": np.random.randn(hidden_dim1, hidden_dim2).astype(np.float32)
            * math.sqrt(2.0 / hidden_dim1),
            "b2": np.zeros((1, hidden_dim2), dtype=np.float32),
            "W3": np.random.randn(hidden_dim2, output_dim).astype(np.float32)
            * math.sqrt(2.0 / hidden_dim2),
            "b3": np.zeros((1, output_dim), dtype=np.float32),
        }

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        z1 = x @ self.params["W1"] + self.params["b1"]
        a1 = np.maximum(0.0, z1)

        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = np.maximum(0.0, z2)

        logits = a2 @ self.params["W3"] + self.params["b3"]

        cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "logits": logits}
        return logits, cache

    def loss_and_gradients(
        self, x: np.ndarray, y_true: np.ndarray
    ) -> Tuple[float, Dict[str, np.ndarray], np.ndarray]:
        logits, cache = self.forward(x)
        probs = softmax(logits)
        batch_size = x.shape[0]
        y_one_hot = one_hot(y_true, probs.shape[1])

        loss = -np.sum(y_one_hot * np.log(probs + 1e-12)) / batch_size

        dlogits = (probs - y_one_hot) / batch_size
        grads: Dict[str, np.ndarray] = {}

        grads["W3"] = cache["a2"].T @ dlogits
        grads["b3"] = np.sum(dlogits, axis=0, keepdims=True)

        da2 = dlogits @ self.params["W3"].T
        dz2 = da2 * (cache["z2"] > 0)
        grads["W2"] = cache["a1"].T @ dz2
        grads["b2"] = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (cache["z1"] > 0)
        grads["W1"] = cache["x"].T @ dz1
        grads["b1"] = np.sum(dz1, axis=0, keepdims=True)

        return float(loss), grads, probs

    def update(self, grads: Dict[str, np.ndarray], learning_rate: float) -> None:
        for name, grad in grads.items():
            self.params[name] -= learning_rate * grad

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(x)
        return np.argmax(logits, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(x)
        return softmax(logits)


# Training module:
# Perform mini-batch optimization and track metrics on training and validation sets.
def iterate_minibatches(
    x: np.ndarray, y: np.ndarray, batch_size: int, seed: int, epoch: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + epoch)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


def evaluate_model(model: ThreeLayerMLP, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    logits, _ = model.forward(x)
    probs = softmax(logits)
    y_one_hot = one_hot(y, probs.shape[1])
    loss = -np.sum(y_one_hot * np.log(probs + 1e-12)) / x.shape[0]
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y, preds)
    return float(loss), float(acc)


def train_model(
    model: ThreeLayerMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    history: List[Dict[str, float]] = []
    best_state: Dict[str, object] = {
        "epoch": 0,
        "val_accuracy": -1.0,
        "params": copy.deepcopy(model.params),
    }

    for epoch in range(1, config.epochs + 1):
        batch_losses: List[float] = []

        for x_batch, y_batch in iterate_minibatches(
            x_train, y_train, config.batch_size, config.seed, epoch
        ):
            loss, grads, _ = model.loss_and_gradients(x_batch, y_batch)
            model.update(grads, config.learning_rate)
            batch_losses.append(loss)

        train_loss, train_acc = evaluate_model(model, x_train, y_train)
        val_loss, val_acc = evaluate_model(model, x_val, y_val)

        row = {
            "epoch": epoch,
            "batch_loss_mean": float(np.mean(batch_losses)),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(row)
        if val_acc > float(best_state["val_accuracy"]):
            best_state = {
                "epoch": epoch,
                "val_accuracy": val_acc,
                "params": copy.deepcopy(model.params),
            }
        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    return history, best_state


# Output module:
# Save metrics, confusion matrix, sample predictions and model weights for submission.
def save_history(history: List[Dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_confusion_matrix(matrix: np.ndarray, class_names: List[str], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/pred"] + class_names)
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name] + row)


def save_class_distribution(
    class_names: List[str],
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class_name", "total", "train", "val", "test"])
        for class_id, class_name in enumerate(class_names):
            total = int(np.sum(y == class_id))
            train_count = int(np.sum(y[train_idx] == class_id))
            val_count = int(np.sum(y[val_idx] == class_id))
            test_count = int(np.sum(y[test_idx] == class_id))
            writer.writerow([class_name, total, train_count, val_count, test_count])


def save_sample_predictions(
    image_paths: List[Path],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str],
    output_path: Path,
    limit: int = 40,
) -> None:
    confidences = probabilities[np.arange(len(y_pred)), y_pred]
    sorted_indices = np.argsort(-confidences)[:limit]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "true_label", "pred_label", "confidence", "is_correct"])
        for idx in sorted_indices:
            writer.writerow(
                [
                    str(image_paths[idx]),
                    class_names[int(y_true[idx])],
                    class_names[int(y_pred[idx])],
                    f"{float(confidences[idx]):.6f}",
                    int(y_true[idx] == y_pred[idx]),
                ]
            )


def save_prediction_grid(
    image_paths: List[Path],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
    limit: int = 16,
) -> None:
    chosen_paths = image_paths[:limit]
    chosen_true = y_true[:limit]
    chosen_pred = y_pred[:limit]

    thumb_size = 96
    caption_height = 26
    cols = 4
    rows = math.ceil(len(chosen_paths) / cols)
    canvas = Image.new(
        "RGB",
        (cols * thumb_size, rows * (thumb_size + caption_height)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)

    for idx, image_path in enumerate(chosen_paths):
        with Image.open(image_path) as image:
            thumb = image.convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)

        row = idx // cols
        col = idx % cols
        x = col * thumb_size
        y = row * (thumb_size + caption_height)
        canvas.paste(thumb, (x, y))

        correct = chosen_true[idx] == chosen_pred[idx]
        text = f"T:{class_names[int(chosen_true[idx])][:7]} P:{class_names[int(chosen_pred[idx])][:7]}"
        text_color = (0, 128, 0) if correct else (200, 0, 0)
        draw.text((x + 2, y + thumb_size + 6), text, fill=text_color)

    canvas.save(output_path)


def save_metrics_json(metrics: Dict[str, object], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)


def save_model(
    model: ThreeLayerMLP, mean: np.ndarray, std: np.ndarray, class_names: List[str], output_path: Path
) -> None:
    np.savez_compressed(
        output_path,
        W1=model.params["W1"],
        b1=model.params["b1"],
        W2=model.params["W2"],
        b2=model.params["b2"],
        W3=model.params["W3"],
        b3=model.params["b3"],
        mean=mean,
        std=std,
        class_names=np.array(class_names),
    )


def run_experiment(config: Config) -> None:
    set_seed(config.seed)
    ensure_dir(config.output_dir)

    print("Loading dataset...")
    x, y, class_names, image_paths = load_dataset(config)
    print(f"Loaded {x.shape[0]} images with input_dim={x.shape[1]} and {len(class_names)} classes.")

    train_idx, val_idx, test_idx = stratified_split(
        y, config.train_ratio, config.val_ratio, config.seed
    )
    x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    x_train, x_val, x_test, mean, std = normalize_by_train(x_train, x_val, x_test)

    model = ThreeLayerMLP(
        input_dim=x_train.shape[1],
        hidden_dim1=config.hidden_dim1,
        hidden_dim2=config.hidden_dim2,
        output_dim=len(class_names),
    )

    history, best_state = train_model(model, x_train, y_train, x_val, y_val, config)
    model.params = copy.deepcopy(best_state["params"])

    train_loss, train_acc = evaluate_model(model, x_train, y_train)
    val_loss, val_acc = evaluate_model(model, x_val, y_val)
    test_loss, test_acc = evaluate_model(model, x_test, y_test)
    test_probs = model.predict_proba(x_test)
    test_preds = np.argmax(test_probs, axis=1)

    matrix = confusion_matrix(y_test, test_preds, len(class_names))
    report_rows, macro_precision, macro_recall, macro_f1 = classification_report_from_confusion(
        matrix, class_names
    )

    save_history(history, config.output_dir / "training_history.csv")
    save_confusion_matrix(matrix, class_names, config.output_dir / "confusion_matrix.csv")
    save_class_distribution(
        class_names, y, train_idx, val_idx, test_idx, config.output_dir / "class_distribution.csv"
    )
    save_sample_predictions(
        [image_paths[idx] for idx in test_idx],
        y_test,
        test_preds,
        test_probs,
        class_names,
        config.output_dir / "sample_predictions.csv",
    )
    save_prediction_grid(
        [image_paths[idx] for idx in test_idx],
        y_test,
        test_preds,
        class_names,
        config.output_dir / "prediction_grid.jpg",
    )
    save_model(model, mean, std, class_names, config.output_dir / "model_weights.npz")

    metrics = {
        "config": {
            "image_size": config.image_size,
            "hidden_dim1": config.hidden_dim1,
            "hidden_dim2": config.hidden_dim2,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "seed": config.seed,
            "max_per_class": config.max_per_class,
        },
        "dataset": {
            "num_samples": int(x.shape[0]),
            "num_classes": len(class_names),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
            "class_names": class_names,
        },
        "metrics": {
            "best_epoch": int(best_state["epoch"]),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        },
        "per_class_metrics": report_rows,
    }
    save_metrics_json(metrics, config.output_dir / "metrics.json")

    print(json.dumps(metrics["metrics"], indent=2, ensure_ascii=False))
    print(f"Outputs saved to: {config.output_dir}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="从零开始构建三层神经网络分类器，实现地表覆盖图像分类。"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("EuroSAT_RGB"),
        help="EuroSAT_RGB 数据集目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="训练输出目录。",
    )
    parser.add_argument("--image-size", type=int, default=16, help="图像缩放尺寸。")
    parser.add_argument("--hidden-dim1", type=int, default=256, help="第一隐藏层维度。")
    parser.add_argument("--hidden-dim2", type=int, default=128, help="第二隐藏层维度。")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="学习率。")
    parser.add_argument("--batch-size", type=int, default=256, help="批大小。")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="每个类别最多读取多少张图像，0 表示使用全部图像。",
    )
    args = parser.parse_args()

    return Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    run_experiment(parse_args())
