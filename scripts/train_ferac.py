import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.emotion.facial_emotion_model import FacialEmotionCNN


CLASSES = list(FacialEmotionCNN.CLASSES)


def build_loaders(data_dir, batch_size, validation_fraction=0.2, seed=42):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    root = Path(data_dir)
    train_base = datasets.ImageFolder(root / "train", transform=train_transform)
    train_eval_base = datasets.ImageFolder(root / "train", transform=eval_transform)
    test = datasets.ImageFolder(root / "test", transform=eval_transform)

    expected = set(CLASSES)
    if set(train_base.classes) != expected or set(test.classes) != expected:
        raise ValueError(
            f"FERAC classes do not match expected classes. "
            f"train={train_base.classes}, test={test.classes}, expected={CLASSES}"
        )

    targets = np.asarray(train_base.targets)
    indices = np.arange(len(train_base))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=targets,
    )

    train = Subset(train_base, train_idx.tolist())
    validation = Subset(train_eval_base, val_idx.tolist())

    return (
        train,
        validation,
        test,
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    return report, matrix.tolist()


def main():
    parser = argparse.ArgumentParser(description="Train Chuwi's FERAC facial-emotion baseline")
    parser.add_argument("--data-dir", required=True, help="Path to the extracted FERAC Dataset directory")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="models/ferac")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the untouched FERAC test split once training/configuration is locked.",
    )
    args = parser.parse_args()

    (
        train_set,
        validation_set,
        test_set,
        train_loader,
        validation_loader,
        test_loader,
    ) = build_loaders(
        args.data_dir,
        args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FacialEmotionCNN(len(CLASSES)).to(device)

    train_targets = torch.tensor([train_set.dataset.targets[i] for i in train_set.indices])
    counts = torch.bincount(train_targets, minlength=len(CLASSES)).float()
    weights = counts.sum() / (len(CLASSES) * counts.clamp_min(1.0))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        report, matrix = evaluate(model, validation_loader, device)
        macro_f1 = report["macro avg"]["f1-score"]
        train_loss = running_loss / len(train_set)
        print(f"epoch={epoch:02d} loss={train_loss:.4f} val_macro_f1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "best_validation_macro_f1": best_f1,
                    "seed": args.seed,
                    "selection_split": "validation",
                },
                output_dir / "ferac_cnn.pt",
            )
            (output_dir / "validation_metrics.json").write_text(
                json.dumps({"classification_report": report, "confusion_matrix": matrix}, indent=2),
                encoding="utf-8",
            )

    print(f"best_validation_macro_f1={best_f1:.4f}")
    print(f"checkpoint={output_dir / 'ferac_cnn.pt'}")

    if args.evaluate_test:
        checkpoint = torch.load(output_dir / "ferac_cnn.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_report, test_matrix = evaluate(model, test_loader, device)
        (output_dir / "test_metrics.json").write_text(
            json.dumps({"classification_report": test_report, "confusion_matrix": test_matrix}, indent=2),
            encoding="utf-8",
        )
        print(f"test_macro_f1={test_report['macro avg']['f1-score']:.4f}")
    else:
        print("Test split was NOT evaluated. Use --evaluate-test only for the final locked evaluation.")


if __name__ == "__main__":
    main()
