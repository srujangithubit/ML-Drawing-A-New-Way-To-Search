from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np


def read_classes(path: str | Path) -> list[str]:
    class_path = Path(path)
    if not class_path.exists():
        raise FileNotFoundError(f"Class list not found: {class_path}")
    labels = [line.strip() for line in class_path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label and not label.startswith("#")]
    if len(labels) < 2:
        raise ValueError("At least two class names are required.")
    if len(set(labels)) != len(labels):
        raise ValueError("Class names must be unique.")
    return labels


def class_filename(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"{slug}.npy"


def load_quickdraw_subset(
    data_dir: str | Path,
    classes: Iterable[str],
    samples_per_class: int | None = 4000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    data_path = Path(data_dir)
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for class_index, class_name in enumerate(classes):
        file_path = data_path / class_filename(class_name)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing {file_path}. Run `python scripts/download_data.py` first."
            )
        bitmap = np.load(file_path, mmap_mode="r")
        if bitmap.ndim == 2 and bitmap.shape[1] == 784:
            bitmap = bitmap.reshape(-1, 28, 28)
        elif bitmap.ndim == 3 and bitmap.shape[1:] == (28, 28):
            pass
        else:
            raise ValueError(
                f"Unexpected bitmap shape {bitmap.shape} in {file_path}; expected (N, 784) or (N, 28, 28)."
            )

        total = len(bitmap)
        take = total if samples_per_class is None else min(total, samples_per_class)
        if take < 2:
            raise ValueError(f"Not enough drawings found for class '{class_name}'.")
        indices = rng.choice(total, size=take, replace=False)
        images.append(np.asarray(bitmap[indices], dtype=np.uint8))
        labels.append(np.full(take, class_index, dtype=np.int64))

    X = np.concatenate(images, axis=0)
    y = np.concatenate(labels, axis=0)
    order = rng.permutation(len(y))
    return X[order], y[order]
