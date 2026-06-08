# -*- coding: utf-8 -*-
"""
pif/evaluator.py
================
Metrics aggregation, summary table generation, and CSV export.

The Evaluator takes raw per-fold result lists (as produced by
LOSOClassifierPipeline or KFoldClassifierPipeline) and computes:
  - Mean ± Std across folds for each model
  - A ranked summary DataFrame
  - Optional per-model CSV export
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from .config import PifConfig


class Evaluator:
    """Aggregate fold results and generate summary reports.

    Parameters
    ----------
    config : PifConfig
    """

    METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1_Macro"]

    def __init__(self, config: PifConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def summarize(
        self,
        results: Dict[str, List[Dict]],
        save_dir: str | None = None,
    ) -> pd.DataFrame:
        """Produce a ranked summary table across all models.

        Parameters
        ----------
        results : dict
            Output of LOSOClassifierPipeline.run() or KFoldClassifierPipeline.run().
            Keys: 'ml', 'fcnn', 'cnn'.
        save_dir : str, optional
            If provided, save the summary CSV here.

        Returns
        -------
        summary_df : pd.DataFrame
            Columns: Model, Accuracy, Precision, Recall, F1_Macro
            Values:  "mean ± std" strings, sorted by F1_Macro descending.
        """
        all_records = []

        # ML models
        if results.get("ml"):
            ml_df = pd.DataFrame(results["ml"])
            for model_name, grp in ml_df.groupby("Model"):
                all_records.append(self._agg_row(model_name, grp))

        # FCNN
        if results.get("fcnn"):
            fcnn_df = pd.DataFrame(results["fcnn"])
            fcnn_df = fcnn_df.rename(
                columns={
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1_macro": "F1_Macro",
                }
            )
            all_records.append(self._agg_row("FCNN", fcnn_df))

        # CNN
        if results.get("cnn"):
            cnn_df = pd.DataFrame(results["cnn"])
            cnn_df = cnn_df.rename(
                columns={
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1_macro": "F1_Macro",
                }
            )
            all_records.append(self._agg_row("CNN", cnn_df))

        summary_df = (
            pd.DataFrame(all_records)
            .sort_values("_f1_sort", ascending=False)
            .drop(columns=["_f1_sort"])
            .reset_index(drop=True)
        )

        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(summary_df.to_string(index=False))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, "summary.csv")
            summary_df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"\n[Evaluator] Summary saved → {path}")

        return summary_df

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _agg_row(self, model_name: str, df: pd.DataFrame) -> Dict:
        row = {"Model": model_name}
        f1_mean = 0.0
        for col in self.METRIC_COLS:
            if col in df.columns:
                mu = df[col].mean()
                sd = df[col].std()
                row[col] = f"{mu:.4f} ± {sd:.4f}"
                if col == "F1_Macro":
                    f1_mean = mu
        row["_f1_sort"] = f1_mean
        return row
