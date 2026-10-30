"""Use a best.pt checkpoint on a compatible per-subject NPZ (X/y/p)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import PGCSCL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="NPZ with X=[N,C,5] and p=[5]")
    parser.add_argument("--output", default="predictions.npz")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    scaler = {k: np.asarray(v, dtype=np.float32) for k, v in ckpt["scaler"].items()}
    with np.load(args.input, allow_pickle=False) as data:
        x = np.asarray(data["X"], dtype=np.float32)
        p = np.asarray(data["p"], dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (cfg["channels"], 5):
        raise ValueError(f"Input X must be [N,{cfg['channels']},5]")
    if p.shape == (5,):
        p = np.broadcast_to(p, (len(x), 5)).copy()
    if p.shape != (len(x), 5):
        raise ValueError("p must be [5] or [N,5]")
    x = (x - scaler["x_mean"]) / scaler["x_std"]
    p = (p - scaler["p_mean"]) / scaler["p_std"]
    dataset = TensorDataset(torch.from_numpy(x.copy()), torch.from_numpy(p.copy()))
    loader = DataLoader(dataset, batch_size=args.batch_size)
    device = torch.device(args.device)
    model = PGCSCL(cfg["channels"], cfg["classes"], cfg["bands"],
                   **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    probabilities, gates = [], []
    with torch.no_grad():
        for bx, bp in loader:
            out = model(bx.to(device), bp.to(device), with_projection=False)
            probabilities.append(out["logits"].softmax(dim=-1).cpu().numpy())
            gates.append(out["gate"].cpu().numpy())
    prob = np.concatenate(probabilities)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, predictions=prob.argmax(1),
                        probabilities=prob, view_gates=np.concatenate(gates))
    print(f"Saved {len(prob)} predictions to {args.output}")


if __name__ == "__main__":
    main()
