"""Per-subject NPZ dataset, leak-free fold statistics, and LOSO partitions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EEGCollection:
    x: np.ndarray                   # [N, C, 5]
    y: np.ndarray                   # [N]
    subject: np.ndarray             # [N] integer index
    personality: np.ndarray         # [N, 5] raw subject personality
    names: list[str]                # index -> stable subject name

    @property
    def subjects(self) -> np.ndarray:
        return np.arange(len(self.names), dtype=np.int64)


def load_subject_npz(directory: str | Path, channels: int, classes: int) -> EEGCollection:
    """Each <subject-id>.npz must contain X [N,C,5], y [N], p [5]."""
    paths = sorted(Path(directory).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No per-subject .npz files in {directory}")
    xs, ys, ss, ps, names = [], [], [], [], []
    for sid, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            if not {"X", "y", "p"}.issubset(data.files):
                raise ValueError(f"{path}: requires keys X, y, p")
            x = np.asarray(data["X"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.int64).reshape(-1)
            p = np.asarray(data["p"], dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (channels, 5) or x.shape[0] != len(y):
            raise ValueError(f"{path}: expected X=[N,{channels},5] and y=[N]")
        if p.shape == (len(y), 5):
            if not np.allclose(p, p[0], atol=1e-5):
                raise ValueError(f"{path}: personality must be constant per subject")
            p = p[0]
        if p.shape != (5,):
            raise ValueError(f"{path}: expected p=[5] in O,C,E,A,N order")
        if x.shape[0] == 0 or (y < 0).any() or (y >= classes).any():
            raise ValueError(f"{path}: empty subject or labels outside [0,{classes-1}]")
        if not np.isfinite(x).all() or not np.isfinite(p).all():
            raise ValueError(f"{path}: inputs contain NaN or infinity")
        xs.append(x)
        ys.append(y)
        ss.append(np.full(len(y), sid, dtype=np.int64))
        ps.append(np.repeat(p[None], len(y), axis=0))
        names.append(path.stem)
    return EEGCollection(np.concatenate(xs), np.concatenate(ys),
                         np.concatenate(ss), np.concatenate(ps), names)


def loso_partition(collection: EEGCollection, fold: int, validation_subjects: int,
                   partition_seed: int = 2026) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source-only deterministic subject-wise validation, unchanged over model seeds."""
    subjects = collection.subjects
    if fold < 0 or fold >= len(subjects):
        raise ValueError(f"fold index must be 0..{len(subjects)-1}")
    remaining = subjects[subjects != fold]
    if validation_subjects < 1 or len(remaining) <= validation_subjects:
        raise ValueError("Not enough source subjects for a validation split")
    rng = np.random.default_rng(partition_seed + fold)
    val_subjects = rng.choice(remaining, size=validation_subjects, replace=False)
    train_subjects = np.setdiff1d(remaining, val_subjects)
    return (np.flatnonzero(np.isin(collection.subject, train_subjects)),
            np.flatnonzero(np.isin(collection.subject, val_subjects)),
            np.flatnonzero(collection.subject == fold))


@dataclass
class FoldScaler:
    x_mean: np.ndarray             # [1, C, 5]
    x_std: np.ndarray              # [1, C, 5]
    p_mean: np.ndarray             # [1, 5]
    p_std: np.ndarray              # [1, 5]

    @classmethod
    def fit(cls, collection: EEGCollection, train_idx: np.ndarray) -> "FoldScaler":
        """Fit DE on train samples; fit personality on UNIQUE TRAIN SUBJECTS."""
        if not len(train_idx):
            raise ValueError("Empty training fold")
        x = collection.x[train_idx].astype(np.float64)
        ids = np.unique(collection.subject[train_idx])
        subject_ps = np.stack([
            collection.personality[train_idx[np.flatnonzero(collection.subject[train_idx] == s)[0]]]
            for s in ids
        ]).astype(np.float64)
        x_mean = x.mean(axis=0, keepdims=True)
        x_std = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
        p_mean = subject_ps.mean(axis=0, keepdims=True)
        p_std = np.maximum(subject_ps.std(axis=0, keepdims=True), 1e-6)
        return cls(x_mean.astype(np.float32), x_std.astype(np.float32),
                   p_mean.astype(np.float32), p_std.astype(np.float32))

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.x_mean) / self.x_std).astype(np.float32)

    def transform_p(self, p: np.ndarray) -> np.ndarray:
        return ((p - self.p_mean) / self.p_std).astype(np.float32)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"x_mean": self.x_mean, "x_std": self.x_std,
                "p_mean": self.p_mean, "p_std": self.p_std}


class FoldDataset(Dataset):
    """Cached fold subset with train-only standardized inputs."""

    def __init__(self, collection: EEGCollection, indices: np.ndarray, scaler: FoldScaler):
        self.x = torch.from_numpy(scaler.transform_x(collection.x[indices]).copy())
        self.y = torch.from_numpy(collection.y[indices].copy())
        self.subject = torch.from_numpy(collection.subject[indices].copy())
        self.personality = torch.from_numpy(scaler.transform_p(collection.personality[indices]).copy())

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.subject[index], self.personality[index]
