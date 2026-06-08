# -*- coding: utf-8 -*-
"""
pif/classifiers.py
==================
ML and Deep Learning classifier wrappers with LOSO and k-fold CV support.

Classifiers mirrored from emjohann/anatomylearning (loso.py / k_fold.py):
  Classical ML : SVM, XGBoost, Random Forest, Logistic Regression, KNN, Naive Bayes
  Deep Learning: FCNN (MLP with BatchNorm + Dropout), CNN (1D Conv)

Both LOSOClassifierPipeline and KFoldClassifierPipeline follow the same
interface: .run(X, y, ...) → list[dict] of per-fold results.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from .config import PifConfig


# ======================================================================== #
# Deep Learning Models
# ======================================================================== #

class _FCNNClassifier(nn.Module):
    """Fully-connected neural network for binary CL classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CNNClassifier(nn.Module):
    """1-D convolutional network for temporal physiological signal patterns."""

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        conv_out = 64 * (input_dim // 4)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.unsqueeze(1))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ======================================================================== #
# Shared training utilities
# ======================================================================== #

def _train_dl_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Dict[str, Any]:
    """Train a PyTorch model and return best-epoch metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    Xv = torch.FloatTensor(X_test).to(device)
    yv = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    best = dict(epoch=0, accuracy=0.0, precision=0.0, recall=0.0, f1_macro=0.0)

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = (model(Xv) >= 0.5).float().cpu().numpy().flatten()
            true = yv.cpu().numpy().flatten()

        f1 = f1_score(true, preds, average="macro", zero_division=0)
        if f1 > best["f1_macro"]:
            best.update(
                epoch=epoch,
                accuracy=accuracy_score(true, preds),
                precision=precision_score(true, preds, zero_division=0),
                recall=recall_score(true, preds, zero_division=0),
                f1_macro=f1,
            )

    return best


def _build_sklearn_models(config: PifConfig) -> Dict[str, Any]:
    rs = config.random_state
    return {
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=rs),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=5, random_state=rs, eval_metric="logloss"
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=rs),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=rs),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
    }


# ======================================================================== #
# LOSO Pipeline
# ======================================================================== #

class LOSOClassifierPipeline:
    """Leave-One-Subject-Out cross-validation pipeline.

    Parameters
    ----------
    config : PifConfig
    """

    def __init__(self, config: PifConfig) -> None:
        self.config = config

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        participant_ids: np.ndarray,
    ) -> Dict[str, List[Dict]]:
        """Run LOSO evaluation across all participants.

        Returns
        -------
        results : dict with keys 'ml', 'fcnn', 'cnn'
            Each value is a list of per-fold result dicts.
        """
        enabled = set(self.config.enabled_classifiers or [])
        ml_models = {
            k: v
            for k, v in _build_sklearn_models(self.config).items()
            if k in enabled
        }

        ml_results, fcnn_results, cnn_results = [], [], []
        unique_pids = np.unique(participant_ids)

        for fold, test_pid in enumerate(unique_pids, 1):
            print(f"\n[LOSO] Fold {fold}/{len(unique_pids)} — test participant: {test_pid}")
            train_mask = participant_ids != test_pid
            test_mask = participant_ids == test_pid

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_te, y_te = X[test_mask], y[test_mask]

            # --- Classical ML ---
            for name, model in ml_models.items():
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                ml_results.append(self._score(name, test_pid, y_te, y_pred))

            # --- FCNN ---
            if "FCNN" in enabled:
                fcnn = _FCNNClassifier(
                    X_tr.shape[1], self.config.dl_hidden_dim, self.config.dl_dropout
                )
                best = _train_dl_model(
                    fcnn, X_tr, y_tr, X_te, y_te,
                    self.config.dl_epochs, self.config.dl_batch_size,
                    self.config.dl_learning_rate,
                )
                best["Test_Participant"] = test_pid
                best["Model"] = "FCNN"
                fcnn_results.append(best)

            # --- CNN ---
            if "CNN" in enabled:
                cnn = _CNNClassifier(X_tr.shape[1], self.config.dl_dropout)
                best = _train_dl_model(
                    cnn, X_tr, y_tr, X_te, y_te,
                    self.config.dl_epochs, self.config.dl_batch_size,
                    self.config.dl_learning_rate,
                )
                best["Test_Participant"] = test_pid
                best["Model"] = "CNN"
                cnn_results.append(best)

        return {"ml": ml_results, "fcnn": fcnn_results, "cnn": cnn_results}

    def save_results(self, results: Dict, save_dir: str) -> None:
        """Persist per-fold results to CSV files."""
        os.makedirs(save_dir, exist_ok=True)
        for key, records in results.items():
            if not records:
                continue
            df = pd.DataFrame(records)
            path = os.path.join(save_dir, f"LOSO_{key}_results.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[LOSO] Saved → {path}")

    @staticmethod
    def _score(
        model_name: str, test_pid: Any, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict:
        return dict(
            Model=model_name,
            Test_Participant=test_pid,
            Accuracy=accuracy_score(y_true, y_pred),
            Precision=precision_score(y_true, y_pred, zero_division=0),
            Recall=recall_score(y_true, y_pred, zero_division=0),
            F1_Macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        )


# ======================================================================== #
# k-Fold Pipeline
# ======================================================================== #

class KFoldClassifierPipeline:
    """Stratified k-fold cross-validation pipeline.

    Parameters
    ----------
    config : PifConfig
    n_splits : int, optional
        Overrides config.n_splits if provided.
    """

    def __init__(self, config: PifConfig, n_splits: int | None = None) -> None:
        self.config = config
        self.n_splits = n_splits or config.n_splits

    def run(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[Dict]]:
        """Run stratified k-fold evaluation.

        Returns
        -------
        results : dict with keys 'ml', 'fcnn', 'cnn'
        """
        enabled = set(self.config.enabled_classifiers or [])
        ml_models = {
            k: v
            for k, v in _build_sklearn_models(self.config).items()
            if k in enabled
        }

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.config.random_state
        )
        ml_results, fcnn_results, cnn_results = [], [], []

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n[k-Fold] Fold {fold}/{self.n_splits}")
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]

            for name, model in ml_models.items():
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                ml_results.append(LOSOClassifierPipeline._score(name, fold, y_te, y_pred))

            if "FCNN" in enabled:
                fcnn = _FCNNClassifier(
                    X_tr.shape[1], self.config.dl_hidden_dim, self.config.dl_dropout
                )
                best = _train_dl_model(
                    fcnn, X_tr, y_tr, X_te, y_te,
                    self.config.dl_epochs, self.config.dl_batch_size,
                    self.config.dl_learning_rate,
                )
                best["Fold"] = fold
                best["Model"] = "FCNN"
                fcnn_results.append(best)

            if "CNN" in enabled:
                cnn = _CNNClassifier(X_tr.shape[1], self.config.dl_dropout)
                best = _train_dl_model(
                    cnn, X_tr, y_tr, X_te, y_te,
                    self.config.dl_epochs, self.config.dl_batch_size,
                    self.config.dl_learning_rate,
                )
                best["Fold"] = fold
                best["Model"] = "CNN"
                cnn_results.append(best)

        return {"ml": ml_results, "fcnn": fcnn_results, "cnn": cnn_results}

    def save_results(self, results: Dict, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        for key, records in results.items():
            if not records:
                continue
            df = pd.DataFrame(records)
            path = os.path.join(save_dir, f"KFold_{key}_results.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[k-Fold] Saved → {path}")
