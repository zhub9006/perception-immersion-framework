# -*- coding: utf-8 -*-
"""
pif/eeg_cnn.py
==============
EEG-specific 1D Convolutional Neural Network for cognitive load classification.

Architecture and design directly informed by reviewing:
  - Diaz265/EEG-Cognitive-State-Classifier (PyTorch, cnn_model.py)
  - harshitsingh4321/1DCNN-Mental-Workload-Classifier (TF/Keras, STEW dataset)

Key design decisions
--------------------
- Input shape : (batch, n_channels, n_timepoints)
    EEG is treated as a multi-channel 1D temporal signal.
- Three Conv1d blocks with BatchNorm + ReLU + MaxPool for hierarchical
  temporal feature extraction (mirrors Diaz265 architecture).
- AdaptiveAvgPool1d(1) as final spatial aggregation so the model handles
  variable-length input windows without architecture changes.
- Classifier head: Linear → ReLU → Dropout(0.4) → Linear(n_classes)
    Dropout rate 0.4 from Diaz265; suitable for small EEG datasets.

Preprocessing expected upstream (see SignalProcessor / preprocess.py):
  - Bandpass filter 0.5–45 Hz (removes DC drift and high-freq artefacts)
  - Z-score normalization (row-wise or per-channel)
  - Segmentation into fixed-length windows (default 128 timepoints @ 128 Hz = 1 s)

References
----------
- Diaz265/EEG-Cognitive-State-Classifier — https://github.com/Diaz265/EEG-Cognitive-State-Classifier
- harshitsingh4321/1DCNN-Mental-Workload-Classifier — https://github.com/harshitsingh4321/1DCNN-Mental-Workload-Classifier
- STEW Dataset: Lim et al. (2018), IEEE TNSRE
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGCognitiveLoadCNN(nn.Module):
    """1D CNN for binary EEG cognitive load classification.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (e.g. 14 for Emotiv, 32 for BrainProducts).
    n_classes : int
        Number of output classes (default 2: low vs. high load).
    dropout : float
        Dropout probability in the classifier head (default 0.4).

    Example
    -------
    >>> model = EEGCognitiveLoadCNN(n_channels=14, n_classes=2)
    >>> x = torch.randn(16, 14, 128)   # batch=16, 14-ch, 128-pt window
    >>> logits = model(x)              # shape (16, 2)
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_classes: int = 2,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        # -------------------------------------------------------------- #
        # Temporal feature extractor
        # Three Conv1d blocks, each doubling the filter count.
        # Kernel sizes decrease (7 → 5 → 3) to capture coarse → fine
        # temporal patterns — standard practice for EEG workload tasks.
        # -------------------------------------------------------------- #
        self.features = nn.Sequential(
            # Block 1 — coarse temporal patterns (alpha/theta rhythms)
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2 — mid-range patterns (beta oscillations)
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3 — fine-grained patterns
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # → (batch, 256, 1)
        )

        # -------------------------------------------------------------- #
        # Classification head
        # -------------------------------------------------------------- #
        self.classifier = nn.Sequential(
            nn.Flatten(),              # (batch, 256)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_channels, n_timepoints)

        Returns
        -------
        logits : torch.Tensor, shape (batch, n_classes)
        """
        x = self.features(x)
        return self.classifier(x)


# ======================================================================== #
# Convenience training utilities
# ======================================================================== #

def train_one_epoch(
    model: EEGCognitiveLoadCNN,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float = 1.0,
) -> float:
    """Run one training epoch and return mean loss.

    Gradient clipping (default norm=1.0) is applied to stabilise
    training on small EEG datasets — approach from Diaz265/train.py.
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: EEGCognitiveLoadCNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model accuracy on a DataLoader.

    Returns
    -------
    dict with keys: accuracy, correct, total
    """
    model.eval()
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
    }
