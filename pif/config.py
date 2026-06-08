# -*- coding: utf-8 -*-
"""
pif/config.py
=============
Global configuration dataclass for the Perception-Immersion Framework.

All experiment-level parameters live here so that every module receives
a single, validated config object rather than scattered keyword arguments.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PifConfig:
    # ------------------------------------------------------------------ #
    # Data paths
    # ------------------------------------------------------------------ #
    data_path: str = "./data/Combined_Data.csv"
    output_dir: str = "./outputs/"

    # ------------------------------------------------------------------ #
    # Column identifiers
    # ------------------------------------------------------------------ #
    participant_id_column: str = "Participant ID"
    label_column: str = "CL_Total"          # CL_Total | ICL_Average | ECL_Average | CL_All_Average
    feature_prefix: str = "Tx"              # Columns whose name starts with this prefix are features

    # ------------------------------------------------------------------ #
    # Cognitive load binarization
    # ------------------------------------------------------------------ #
    cl_threshold: int = 22                  # Scores <= threshold → class 0 (low), > threshold → class 1 (high)
    cl_label_columns: List[str] = field(
        default_factory=lambda: ["CL_All_Average", "ICL_Average", "ECL_Average"]
    )

    # ------------------------------------------------------------------ #
    # Signal preprocessing
    # ------------------------------------------------------------------ #
    normalize_method: str = "zscore_rowwise"   # "zscore_rowwise" | "minmax" | "none"
    drop_na: bool = True

    # ------------------------------------------------------------------ #
    # Cross-validation
    # ------------------------------------------------------------------ #
    cv_strategy: str = "loso"              # "loso" | "kfold"
    n_splits: int = 5                      # Used only when cv_strategy == "kfold"
    random_state: int = 42

    # ------------------------------------------------------------------ #
    # Deep learning training
    # ------------------------------------------------------------------ #
    dl_epochs: int = 100
    dl_batch_size: int = 32
    dl_learning_rate: float = 1e-3
    dl_hidden_dim: int = 128
    dl_dropout: float = 0.3

    # ------------------------------------------------------------------ #
    # Immersion scoring
    # ------------------------------------------------------------------ #
    # Weights for the composite Perception-Immersion Index (PII)
    pii_weight_questionnaire: float = 0.50   # IPQ / SUS / NASA-TLX composite
    pii_weight_physiological: float = 0.30   # Pupil dilation, GSR
    pii_weight_behavioral: float = 0.20      # Dwell time, interaction rate

    # ------------------------------------------------------------------ #
    # Modality flags
    # ------------------------------------------------------------------ #
    modality: str = "fnirs"                 # "fnirs" | "eeg" | "gsr" | "eye" | "multimodal"
    enabled_classifiers: Optional[List[str]] = None   # None → run all

    def __post_init__(self):
        if self.enabled_classifiers is None:
            self.enabled_classifiers = [
                "SVM", "XGBoost", "Random Forest",
                "Logistic Regression", "KNN", "Naive Bayes",
                "FCNN", "CNN",
            ]
