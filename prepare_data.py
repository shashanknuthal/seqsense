"""Convert the official UCI HAR inertial-signal text files into window arrays."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

SIGNALS = (
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
)


def load_split(root: Path, split: str):
    signal_dir = root / split / "Inertial Signals"
    channels = [np.loadtxt(signal_dir / f"{name}_{split}.txt", dtype=np.float32) for name in SIGNALS]
    x = np.stack(channels, axis=-1)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=np.int64)
    if x.shape[1:] != (128, 9):
        raise ValueError(f"Expected windows shaped [N,128,9], received {x.shape}")
    return x, y, subjects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to extracted 'UCI HAR Dataset' directory")
    parser.add_argument("--output", type=Path, default=Path("data/har.npz"))
    args = parser.parse_args()
    train = load_split(args.dataset_root, "train")
    test = load_split(args.dataset_root, "test")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X_train=train[0], y_train=train[1], subj_train=train[2],
                        X_test=test[0], y_test=test[1], subj_test=test[2])
    print(f"Saved {args.output}: train={train[0].shape}, test={test[0].shape}")


if __name__ == "__main__":
    main()
