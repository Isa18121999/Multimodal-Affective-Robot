#!/usr/bin/env python3
"""Reproducible benchmark for Chuwi facial-expression models."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_MAP = {
    "anger": "ANGER",
    "disgust": "DISGUST",
    "fear": "FEAR",
    "happiness": "HAPPY",
    "neutral": "CALM",
    "sadness": "SAD",
    "surprise": "SURPRISE",
}

CLASSES = ("ANGER", "DISGUST", "FEAR", "HAPPY", "CALM", "SAD", "SURPRISE")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(self.root / row["new_path"]).convert("RGB")
        return self.transform(image), CLASSES.index(CLASS_MAP[row["class"]])


def validate_manifest(frame: pd.DataFrame, root: Path) -> None:
    required = {"source", "class", "split", "new_path", "sha256"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame["sha256"].duplicated().any():
        raise ValueError("Duplicate SHA-256 detected in manifest")
    if not set(frame["class"]).issubset(CLASS_MAP):
        raise ValueError("Manifest contains an unsupported emotion class")
    if not set(frame["split"]).issubset({"train", "validation", "test"}):
        raise ValueError("Manifest contains an unsupported split")
    missing_files = [str(root / p) for p in frame["new_path"] if not (root / p).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing dataset files: {missing_files[:5]}")


def build_model(name: str, num_classes: int) -> nn.Module:
    names = {
        "mobilenet_v3_large": "mobilenetv3_large_100",
        "efficientnet_b0": "efficientnet_b0",
        "mobilevit_xxs": "mobilevit_xxs",
        "deit_tiny": "deit_tiny_patch16_224",
    }
    if name not in names:
        raise ValueError(f"Unknown model: {name}")
    try:
        return timm.create_model(names[name], pretrained=True, num_classes=num_classes)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create {name}. Check timm model availability and pretrained weights."
        ) from exc


def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            all_pred.extend(logits.argmax(dim=1).cpu().tolist())
            all_true.extend(labels.tolist())

    elapsed = time.perf_counter() - start
    return {
        "accuracy": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_true, all_pred, average="weighted", zero_division=0),
        "samples": len(all_true),
        "seconds": elapsed,
        "samples_per_second": len(all_true) / elapsed if elapsed else 0.0,
        "confusion_matrix": confusion_matrix(
            all_true, all_pred, labels=range(len(CLASSES))
        ).tolist(),
        "report": classification_report(
            all_true, all_pred,
            labels=range(len(CLASSES)),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_one(model, train_loader, val_loader, device, epochs, lr, weight_decay, patience):
    counts = torch.zeros(len(CLASSES), dtype=torch.float32)
    for _, labels in train_loader:
        counts += torch.bincount(labels, minlength=len(CLASSES)).float()

    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_f1 = -1.0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        val = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch} val_accuracy={val['accuracy']:.4f} "
            f"val_macro_f1={val['macro_f1']:.4f}"
        )

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("No checkpoint was produced")
    model.load_state_dict(best_state)
    return model, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/facial_benchmark"))
    parser.add_argument(
        "--models", nargs="+",
        default=["mobilenet_v3_large", "efficientnet_b0", "mobilevit_xxs", "deit_tiny"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = args.manifest or args.dataset_root / "emotion_manifest.csv"
    frame = pd.read_csv(manifest)
    validate_manifest(frame, args.dataset_root)

    train_frame = frame[frame["split"] == "train"]
    val_frame = frame[frame["split"] == "validation"]
    test_frame = frame[frame["split"] == "test"]
    child_test = test_frame[test_frame["source"].eq("Children_Emotion")]

    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        ManifestDataset(train_frame, args.dataset_root, train_transform),
        shuffle=True, **loader_args
    )
    val_loader = DataLoader(
        ManifestDataset(val_frame, args.dataset_root, eval_transform),
        shuffle=False, **loader_args
    )
    test_loader = DataLoader(
        ManifestDataset(test_frame, args.dataset_root, eval_transform),
        shuffle=False, **loader_args
    )
    child_loader = DataLoader(
        ManifestDataset(child_test, args.dataset_root, eval_transform),
        shuffle=False, **loader_args
    )

    args.output.mkdir(parents=True, exist_ok=True)
    results = []

    for model_name in args.models:
        print(f"\n=== {model_name} on {device} ===")
        model = build_model(model_name, len(CLASSES)).to(device)
        start_train = time.perf_counter()
        model, best_val_f1 = train_one(
            model, train_loader, val_loader, device,
            args.epochs, args.lr, args.weight_decay, args.patience
        )
        training_seconds = time.perf_counter() - start_train

        checkpoint = args.output / f"{model_name}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "classes": CLASSES,
            "seed": args.seed,
            "input_size": 224,
        }, checkpoint)

        test_metrics = evaluate(model, test_loader, device)
        child_metrics = evaluate(model, child_loader, device) if len(child_test) else None
        params = sum(p.numel() for p in model.parameters())

        results.append({
            "model": model_name,
            "parameters": params,
            "best_validation_macro_f1": best_val_f1,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_samples": test_metrics["samples"],
            "children_test_accuracy": child_metrics["accuracy"] if child_metrics else None,
            "children_test_macro_f1": child_metrics["macro_f1"] if child_metrics else None,
            "training_seconds": training_seconds,
            "test_seconds": test_metrics["seconds"],
            "test_samples_per_second": test_metrics["samples_per_second"],
            "checkpoint": str(checkpoint),
        })

        (args.output / f"{model_name}_test.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8"
        )
        if child_metrics:
            (args.output / f"{model_name}_children_test.json").write_text(
                json.dumps(child_metrics, indent=2), encoding="utf-8"
            )

    pd.DataFrame(results).to_csv(args.output / "benchmark_results.csv", index=False)
    (args.output / "run_metadata.json").write_text(json.dumps({
        "seed": args.seed,
        "device": str(device),
        "dataset_root": str(args.dataset_root),
        "manifest": str(manifest),
        "classes": CLASSES,
        "train_samples": len(train_frame),
        "validation_samples": len(val_frame),
        "test_samples": len(test_frame),
        "children_test_samples": len(child_test),
        "models": args.models,
    }, indent=2), encoding="utf-8")

    print(f"\nBenchmark complete: {args.output}")


if __name__ == "__main__":
    main()
