"""Train and evaluate one SeqSense model on prepared UCI HAR windows."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import MODEL_NAMES, build_model

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize(x, mean, std):
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/har.npz"))
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    seed_everything(args.seed)
    device = torch.device(args.device)

    with np.load(args.data) as data:
        x_train, y_train = data["X_train"], data["y_train"]
        x_test, y_test = data["X_test"], data["y_test"]
    mean, std = x_train.mean(axis=(0, 1)), x_train.std(axis=(0, 1))
    x_train = normalize(x_train, mean, std)
    x_test = normalize(x_test, mean, std)
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
                            batch_size=args.batch_size)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_state, stale = -1.0, None, 0
    start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        val_loss, predictions, targets = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(yb)
                predictions.extend(logits.argmax(1).cpu().tolist())
                targets.extend(yb.cpu().tolist())
        val_loss /= len(y_val)
        val_acc = accuracy_score(targets, predictions)
        scheduler.step(val_loss)
        print(f"epoch {epoch + 1:02d}: val_loss={val_loss:.4f} val_accuracy={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 8:
                break

    model.load_state_dict(best_state)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for start_idx in range(0, len(x_test), args.batch_size):
            xb = torch.from_numpy(x_test[start_idx:start_idx + args.batch_size]).to(device)
            test_preds.extend(model(xb).argmax(1).cpu().tolist())
    elapsed = time.perf_counter() - start
    model.to("cpu").eval()
    sample = torch.from_numpy(x_test[:1])
    with torch.no_grad():
        for _ in range(10):
            model(sample)
        t0 = time.perf_counter()
        for _ in range(100):
            model(sample)
    latency_ms = (time.perf_counter() - t0) * 1000 / 100
    report = classification_report(y_test, test_preds, labels=range(6), target_names=LABELS,
                                   output_dict=True, zero_division=0)
    metrics = {
        "model": args.model, "accuracy": float(accuracy_score(y_test, test_preds)),
        "macro_f1": float(f1_score(y_test, test_preds, average="macro")),
        "per_class": {name: report[name] for name in LABELS},
        "confusion_matrix": confusion_matrix(y_test, test_preds, labels=range(6)).tolist(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds_including_validation": elapsed,
        "epochs_run": epoch + 1, "cpu_latency_ms_per_window": latency_ms,
        "seed": args.seed, "train_windows": len(x_fit), "validation_windows": len(x_val),
        "test_windows": len(y_test), "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
        "labels": LABELS,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_name": args.model, "labels": LABELS,
                "normalization_mean": mean.tolist(), "normalization_std": std.tolist()},
               args.output_dir / f"{args.model}.pt")
    with (args.output_dir / f"{args.model}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                      "parameters": metrics["parameters"], "latency_ms": latency_ms}, indent=2))


if __name__ == "__main__":
    main()
