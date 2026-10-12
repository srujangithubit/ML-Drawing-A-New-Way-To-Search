from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data import load_quickdraw_subset, read_classes  # noqa: E402
from src.model import CompactCNN  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x = torch.from_numpy(images.astype(np.float32)[:, None, :, :] / 255.0)
    y = torch.from_numpy(labels.astype(np.int64))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_predictions.append(preds)
            all_targets.append(yb.numpy())
    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    return float(accuracy_score(targets, predictions)), targets, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a compact QuickDraw doodle classifier.")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "numpy_bitmap"))
    parser.add_argument("--classes-file", default=str(ROOT / "config" / "classes.txt"))
    parser.add_argument("--samples-per-class", type=int, default=4000, help="Maximum drawings per class; use 0 for all available drawings.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts"))
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    set_seed(args.seed)
    classes = read_classes(args.classes_file)
    sample_cap = None if args.samples_per_class == 0 else args.samples_per_class

    print(f"Loading classes: {', '.join(classes)}")
    try:
        X, y = load_quickdraw_subset(args.data_dir, classes, sample_cap, seed=args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        print("First run: python scripts/download_data.py", file=sys.stderr)
        return 1

    # Reproducible, stratified 80:10:10 split as described in the source paper.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=args.seed, stratify=y_temp
    )

    train_loader = make_loader(X_train, y_train, args.batch_size, True)
    val_loader = make_loader(X_val, y_val, args.batch_size, False)
    test_loader = make_loader(X_test, y_test, args.batch_size, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; examples: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    model = CompactCNN(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    best_val = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(yb)
            seen += len(yb)
        train_loss = running_loss / max(1, seen)
        val_accuracy, _, _ = evaluate(model, val_loader, device)
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), "val_accuracy": val_accuracy})
        print(f"Epoch {epoch:02d}/{args.epochs} - loss: {train_loss:.4f} - val_accuracy: {val_accuracy * 100:.2f}%")
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    test_accuracy, targets, predictions = evaluate(model, test_loader, device)
    report = classification_report(
        targets, predictions, labels=list(range(len(classes))), target_names=classes,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(targets, predictions, labels=list(range(len(classes))))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "classes": classes,
        "seed": args.seed,
        "samples_per_class_cap": sample_cap,
        "sample_counts": {"train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test))},
        "best_validation_accuracy": float(best_val),
        "test_accuracy": float(test_accuracy),
        "classification_report": report,
        "training_history": history,
        "device": str(device),
        "architecture": "CompactCNN: Conv(16)-Pool-Conv(32)-Pool-FC(128)-Dropout-Softmax",
    }
    torch.save(
        {"model_state_dict": best_state, "classes": classes, "metrics": metrics, "input_size": 28},
        output_dir / "model.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Purples", colorbar=False, values_format="d")
    ax.set_title(f"Held-out test confusion matrix (accuracy {test_accuracy * 100:.2f}%)")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([entry["epoch"] for entry in history], [entry["train_loss"] for entry in history], label="Training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax2 = ax.twinx()
    ax2.plot([entry["epoch"] for entry in history], [entry["val_accuracy"] * 100 for entry in history], label="Validation accuracy")
    ax2.set_ylabel("Validation accuracy (%)")
    ax.set_title("Training history")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=180)
    plt.close(fig)

    print(f"\nBest validation accuracy: {best_val * 100:.2f}%")
    print(f"Held-out test accuracy:   {test_accuracy * 100:.2f}%")
    print(f"Saved: {output_dir / 'model.pt'}")
    print(f"Saved: {output_dir / 'metrics.json'}")
    print(f"Saved: {output_dir / 'confusion_matrix.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
