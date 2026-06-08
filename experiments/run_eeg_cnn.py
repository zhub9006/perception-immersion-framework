#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/run_eeg_cnn.py
==========================
End-to-end EEG cognitive load classification experiment using the
EEGCognitiveLoadCNN — drawn directly from reviewing:
  - Diaz265/EEG-Cognitive-State-Classifier  (train.py)
  - harshitsingh4321/1DCNN-Mental-Workload-Classifier  (notebook)

Usage
-----
    python experiments/run_eeg_cnn.py --subject 1 --epochs 20 --lr 0.0005

The script:
  1. Loads EEG via EEGPreprocessor (MNE EEGBCI dataset by default)
  2. Applies class-balanced WeightedRandomSampler (from Diaz265/train.py)
  3. Trains EEGCognitiveLoadCNN with AdamW + StepLR scheduler
  4. Evaluates and saves model weights to outputs/eeg_cnn/
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from pif.eeg_preprocessor import EEGPreprocessor
from pif.eeg_cnn import EEGCognitiveLoadCNN, train_one_epoch, evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EEG CNN cognitive load experiment")
    p.add_argument("--subject", type=int, default=1, help="EEGBCI subject index")
    p.add_argument("--runs", type=int, nargs="+", default=[6, 10])
    p.add_argument("--n-channels", type=int, default=32)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--output-dir", type=str, default="outputs/eeg_cnn")
    return p.parse_args()


def build_balanced_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> DataLoader:
    """Create a DataLoader with class-balanced WeightedRandomSampler.

    Mirrors the class-balancing strategy in Diaz265/train.py to handle
    imbalanced cognitive load label distributions.
    """
    class_counts = np.bincount(y)
    weights = 1.0 / class_counts
    sample_weights = torch.tensor(weights[y], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load and preprocess EEG data
    # ------------------------------------------------------------------ #
    print(f"[1/4] Loading EEG — subject {args.subject}, runs {args.runs}")
    preprocessor = EEGPreprocessor(
        n_channels=args.n_channels,
        window_size=args.window_size,
    )
    X, y = preprocessor.load_eegbci(subject=args.subject, runs=args.runs)

    # Global normalization (additional pass — mirrors Diaz265/train.py)
    X = (X - X.mean()) / (X.std() + 1e-8)

    print(f"    X shape: {X.shape}  |  y shape: {y.shape}")
    print(f"    Class distribution: {np.bincount(y)}")

    # ------------------------------------------------------------------ #
    # 2. Build data loader
    # ------------------------------------------------------------------ #
    print("[2/4] Building class-balanced DataLoader")
    loader = build_balanced_loader(X, y, batch_size=args.batch_size)

    # ------------------------------------------------------------------ #
    # 3. Initialise model, optimizer, scheduler
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[3/4] Training on {device}")

    model = EEGCognitiveLoadCNN(
        n_channels=args.n_channels,
        n_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    # StepLR: halve LR every 5 epochs (from Diaz265/train.py)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    # ------------------------------------------------------------------ #
    # 4. Training loop
    # ------------------------------------------------------------------ #
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, loader, device)
            print(
                f"  Epoch {epoch:3d}/{args.epochs} | "
                f"Loss: {loss:.4f} | "
                f"Acc: {metrics['accuracy']:.4f}"
            )

    # ------------------------------------------------------------------ #
    # 5. Save weights
    # ------------------------------------------------------------------ #
    save_path = os.path.join(args.output_dir, "eeg_cnn_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n[4/4] Model saved → {save_path}")

    final = evaluate(model, loader, device)
    print(f"      Final accuracy: {final['accuracy']:.4f} "
          f"({final['correct']}/{final['total']})")


if __name__ == "__main__":
    main()
