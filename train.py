"""LOSO training: python -m pgcscl.train --config ... --data-dir ..."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import FoldDataset, FoldScaler, load_subject_npz, loso_partition
from .losses import supervised_contrastive_loss
from .model import PGCSCL
from .sampler import SubjectClassBalancedBatchSampler
from .utils import read_config, save_json, seed_everything, subject_level_summary


def evaluation(model: PGCSCL, loader: DataLoader, device: torch.device,
               num_classes: int) -> dict:
    """Deterministic evaluation: no projection head, no target-domain fitting."""
    model.eval()
    targets, predictions = [], []
    total_ce = total_n = 0
    with torch.no_grad():
        for x, y, _s, p in loader:
            x, y, p = x.to(device), y.to(device), p.to(device)
            outputs = model(x, p, with_projection=False)
            logits = outputs["logits"]
            total_ce += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            total_n += len(y)
            targets.append(y.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    if not total_n:
        raise ValueError("Empty evaluation dataset")
    target = np.concatenate(targets)
    pred = np.concatenate(predictions)
    return {
        "loss": total_ce / total_n,
        "accuracy": float(accuracy_score(target, pred)),
        "macro_f1": float(f1_score(target, pred, labels=list(range(num_classes)),
                                   average="macro", zero_division=0)),
        "samples": total_n,
    }


def train_single_fold(collection, config: dict, fold: int, seed: int,
                      output: Path, device: torch.device,
                      epochs_override: int | None = None,
                      steps_override: int | None = None) -> dict:
    seed_everything(seed)
    train_idx, val_idx, test_idx = loso_partition(
        collection, fold, config["validation_subjects"]
    )
    train_subjects = np.unique(collection.subject[train_idx]).tolist()
    val_subjects = np.unique(collection.subject[val_idx]).tolist()
    assert not (set(train_subjects) & set(val_subjects))
    assert fold not in train_subjects and fold not in val_subjects

    scaler = FoldScaler.fit(collection, train_idx)
    train_dataset = FoldDataset(collection, train_idx, scaler)
    val_dataset = FoldDataset(collection, val_idx, scaler)
    # Construct held-out dataset for final testing; never inspect it in fitting/selection.
    test_dataset = FoldDataset(collection, test_idx, scaler)

    tc = config["train"]
    lc = config["loss"]
    epochs = tc["epochs"] if epochs_override is None else epochs_override
    if epochs < 1:
        raise ValueError("epochs must be positive")
    steps = math.ceil(len(train_dataset) / tc["batch_size"])
    if steps_override is not None:
        steps = steps_override
    sampler = SubjectClassBalancedBatchSampler(
        train_dataset.y.numpy(), train_dataset.subject.numpy(),
        subjects_per_batch=tc["subjects_per_batch"],
        classes_per_batch=tc["classes_per_batch"],
        examples_per_cell=tc["examples_per_subject_class"],
        steps_per_epoch=steps, seed=seed,
    )
    implied_batch = (tc["subjects_per_batch"] * tc["classes_per_batch"]
                     * tc["examples_per_subject_class"])
    if implied_batch != tc["batch_size"]:
        raise ValueError(f"Batch composition gives {implied_batch}, config says {tc['batch_size']}")
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=tc["batch_size"], shuffle=False, num_workers=0)

    model = PGCSCL(channels=config["channels"], classes=config["classes"],
                   bands=config["bands"], **config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"],
                                  weight_decay=tc["weight_decay"])

    fold_dir = output / f"fold_{fold:02d}_{collection.names[fold]}" / f"seed_{seed}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_dir / "best.pt"
    log_rows = []
    best_loss = float("inf")
    patience_counter = 0
    best_epoch = -1
    # Paper anneals Gumbel temperature over the predefined 150-epoch schedule.
    anneal_horizon = max(1, tc["epochs"] - 1)
    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)
        progress = min(epoch / anneal_horizon, 1.0)
        graph_tau = tc["gumbel_tau_start"] + progress * (
            tc["gumbel_tau_end"] - tc["gumbel_tau_start"]
        )
        running_total = running_cls = running_cont = running_sparse = 0.0
        n_batches = 0
        for x, y, subjects, p in train_loader:
            x, y = x.to(device), y.to(device)
            subjects, p = subjects.to(device), p.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, p, gumbel_tau=graph_tau)
            cls_loss = nn.functional.cross_entropy(outputs["logits"], y)
            cont_loss = supervised_contrastive_loss(
                outputs["embedding"], y, subjects, p,
                temperature=lc["temperature"],
                personality_temperature=lc["personality_temperature"],
                variant=lc["contrastive_variant"],
            )
            sparse_loss = outputs["sparse_loss"]
            loss = cls_loss + lc["lambda_cont"] * cont_loss + lc["lambda_sparse"] * sparse_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at fold {fold} seed {seed} epoch {epoch}")
            loss.backward()
            optimizer.step()
            running_total += loss.item()
            running_cls += cls_loss.item()
            running_cont += cont_loss.item()
            running_sparse += sparse_loss.item()
            n_batches += 1

        # Only source-validation classification CE: comparators in val set do
        # not have the training batch-composition guarantee. This is a stated
        # interpretation of 'source-validation loss' in the paper.
        val_stats = evaluation(model, val_loader, device, config["classes"])
        entry = {
            "epoch": epoch + 1, "gumbel_tau": graph_tau,
            "train_total": running_total / n_batches,
            "train_ce": running_cls / n_batches,
            "train_pa_scl": running_cont / n_batches,
            "train_l1_graph": running_sparse / n_batches,
            "val_ce": val_stats["loss"], "val_accuracy": val_stats["accuracy"],
        }
        log_rows.append(entry)
        if val_stats["loss"] < best_loss - 1e-8:
            best_loss = val_stats["loss"]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "scaler": {key: value.tolist() for key, value in scaler.as_dict().items()},
                "fold": fold, "seed": seed, "test_subject": collection.names[fold],
                "train_subjects": [collection.names[s] for s in train_subjects],
                "val_subjects": [collection.names[s] for s in val_subjects],
                "best_epoch": best_epoch,
            }, checkpoint)
        else:
            patience_counter += 1
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0 or patience_counter >= tc["patience"]:
            print(f"fold={fold} seed={seed} epoch={epoch+1} "
                  f"train={entry['train_total']:.4f} val_ce={val_stats['loss']:.4f} "
                  f"best={best_loss:.4f} patience={patience_counter}", flush=True)
        if patience_counter >= tc["patience"]:
            break

    with (fold_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    best = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    test_loader = DataLoader(test_dataset, batch_size=tc["batch_size"], shuffle=False)
    result = evaluation(model, test_loader, device, config["classes"])
    result.update({
        "fold": fold, "test_subject": collection.names[fold], "seed": seed,
        "best_epoch": best_epoch, "best_val_ce": best_loss,
        "train_subjects": len(train_subjects), "validation_subjects": len(val_subjects),
        "checkpoint": str(checkpoint),
    })
    save_json(fold_dir / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Paper-based PGC-SCL LOSO reproduction")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="runs/experiment")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--folds", nargs="*", type=int, help="0-based fold indexes; default: all")
    parser.add_argument("--seeds", nargs="*", type=int, help="Default: config [42,43,44]")
    parser.add_argument("--epochs-override", type=int, help="Debug only, do not use for paper benchmark")
    parser.add_argument("--steps-override", type=int, help="Debug only, do not use for paper benchmark")
    args = parser.parse_args()
    config = read_config(args.config)
    collection = load_subject_npz(args.data_dir, config["channels"], config["classes"])
    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device
    device = torch.device(dev)
    folds = list(range(len(collection.names))) if args.folds is None else args.folds
    seeds = config["train"]["seeds"] if args.seeds is None else args.seeds
    if not folds or not seeds:
        raise ValueError("At least one fold and seed are required")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "run_manifest.json", {
        "config": config, "subjects": collection.names, "folds": folds,
        "seeds": seeds, "data_dir": str(args.data_dir), "device": str(device),
        "debug_epochs_override": args.epochs_override,
        "debug_steps_override": args.steps_override,
    })
    results = []
    for fold in folds:
        for seed in seeds:
            result = train_single_fold(collection, config, fold, seed, out, device,
                                       args.epochs_override, args.steps_override)
            results.append(result)
            print(f"TEST fold={fold}, seed={seed}: accuracy={result['accuracy']:.4f} "
                  f"macro_f1={result['macro_f1']:.4f}", flush=True)
    with (out / "fold_seed_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = subject_level_summary(results)
    summary["complete_loso"] = len(set(folds)) == len(collection.names)
    summary["expected_seeds_per_subject"] = len(seeds)
    save_json(out / "summary.json", summary)
    print("Completed. Final subject-wise summary:", summary, flush=True)


if __name__ == "__main__":
    main()
