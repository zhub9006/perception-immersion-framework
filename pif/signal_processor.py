# -*- coding: utf-8 -*-
"""
pif/signal_processor.py
========================
Raw signal ingestion, cleaning, and row-wise z-score normalization.

Design notes
------------
- Mirrors the preprocessing approach in emjohann/anatomylearning (loso.py):
    1. Load CSV
    2. Drop irrelevant metadata columns (kept flexible via config)
    3. Drop NaN rows
    4. Round CL label columns to integers
    5. Apply row-wise (per-sample) z-score normalization to feature matrix
    6. Replace any residual NaN/Inf with 0

The row-wise strategy is intentional for neurophysiological data: it removes
inter-subject amplitude differences while preserving intra-sample patterns.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from .config import PifConfig

warnings.filterwarnings("ignore")


class SignalProcessor:
    """Load, clean, and normalize a tabular physiological signal dataset.

    Parameters
    ----------
    config : PifConfig
        Experiment configuration object.
    """

    def __init__(self, config: PifConfig) -> None:
        self.config = config
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load_and_preprocess(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full pipeline: load → clean → normalize.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
            Normalized feature matrix.
        y : np.ndarray, shape (n_samples,)
            Raw (unbinarized) label values.
        participant_ids : np.ndarray, shape (n_samples,)
            Participant identifier per sample (for LOSO splitting).
        """
        df = self._load(self.config.data_path)
        df = self._clean(df)
        X, y, participant_ids = self._split_features_labels(df)
        X = self._normalize(X)
        return X, y, participant_ids

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load(self, path: str) -> pd.DataFrame:
        """Load CSV from disk."""
        df = pd.read_csv(path)
        print(f"[SignalProcessor] Loaded {len(df)} rows × {df.shape[1]} cols from '{path}'")
        self._df = df
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop NaN rows and round CL label columns."""
        if self.config.drop_na:
            before = len(df)
            df = df.dropna().reset_index(drop=True)
            print(f"[SignalProcessor] Dropped {before - len(df)} NaN rows → {len(df)} remaining")

        # Round CL label columns to integers
        for col in self.config.cl_label_columns:
            if col in df.columns:
                df[col] = np.round(df[col]).astype(int)

        return df

    def _split_features_labels(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Separate feature columns, label column, and participant IDs."""
        feature_cols = [c for c in df.columns if c.startswith(self.config.feature_prefix)]
        if not feature_cols:
            raise ValueError(
                f"No feature columns found with prefix '{self.config.feature_prefix}'. "
                f"Available columns: {list(df.columns)}"
            )

        X = df[feature_cols].values.astype(float)
        y = df[self.config.label_column].values

        pid_col = self.config.participant_id_column
        if pid_col not in df.columns:
            raise ValueError(f"Participant ID column '{pid_col}' not found in dataset.")
        participant_ids = df[pid_col].values

        print(
            f"[SignalProcessor] Features: {len(feature_cols)} cols | "
            f"Label: '{self.config.label_column}' | "
            f"Participants: {len(np.unique(participant_ids))}"
        )
        return X, y, participant_ids

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize feature matrix according to config.normalize_method."""
        method = self.config.normalize_method

        if method == "zscore_rowwise":
            X = self._zscore_rowwise(X)
        elif method == "minmax":
            X = self._minmax(X)
        elif method == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize_method: '{method}'")

        # Sanitize
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            f"[SignalProcessor] Normalization '{method}' applied. "
            f"Mean={X.mean():.4f}, Std={X.std():.4f}"
        )
        return X

    @staticmethod
    def _zscore_rowwise(X: np.ndarray) -> np.ndarray:
        """Per-sample z-score normalization (row-wise).

        For each sample i:  X[i] = (X[i] - mean(X[i])) / std(X[i])
        Samples with zero std are mean-subtracted only.
        """
        X_norm = np.zeros_like(X, dtype=float)
        for i in range(X.shape[0]):
            mu = X[i].mean()
            sigma = X[i].std()
            X_norm[i] = (X[i] - mu) / sigma if sigma > 0 else X[i] - mu
        return X_norm

    @staticmethod
    def _minmax(X: np.ndarray) -> np.ndarray:
        """Global min-max normalization to [0, 1]."""
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        denom = X_max - X_min
        denom[denom == 0] = 1.0
        return (X - X_min) / denom
