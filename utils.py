"""Seed control, reporting, and configuration."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import t as student_t


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Many CUDA ops are deterministic under these flags, although some
    # platform-specific operations may remain nondeterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def subject_level_summary(rows: list[dict]):
    """Average independent seeds within each held-out subject before CI."""
    fold_data = {}
    for row in rows:
        fold_data.setdefault(int(row["fold"]), []).append(row)
    averaged = []
    for fold, scores in sorted(fold_data.items()):
        averaged.append({
            "fold": fold,
            "subject": scores[0]["test_subject"],
            "seeds": len(scores),
            "accuracy": float(np.mean([r["accuracy"] for r in scores])),
            "macro_f1": float(np.mean([r["macro_f1"] for r in scores])),
        })
    acc = np.array([s["accuracy"] for s in averaged])
    f1 = np.array([s["macro_f1"] for s in averaged])
    n = len(acc)
    std = float(acc.std(ddof=1)) if n > 1 else 0.0
    margin = float(student_t.ppf(0.975, df=n-1) * std / np.sqrt(n)) if n > 1 else 0.0
    mean = float(acc.mean())
    return {
        "n_test_subjects": n,
        "accuracy_mean_percent": mean * 100,
        "accuracy_std_percent": std * 100,
        "accuracy_95ci_percent": [(mean - margin) * 100, (mean + margin) * 100],
        "macro_f1_mean_percent": float(f1.mean()) * 100,
        "per_subject": averaged,
    }
