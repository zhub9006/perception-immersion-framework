# -*- coding: utf-8 -*-
"""
pif/cognitive_load.py
=====================
Cognitive load scoring, threshold-based binarization, and distribution
analysis utilities.

Mirrors the binarization logic from emjohann/anatomylearning (loso.py):
    - CL scores are rounded to integers
    - A single threshold T splits samples into binary classes:
        class 0 (low load):  score <= T
        class 1 (high load): score >  T
    - Default T = 22 (validated for ICL/ECL subscales in the fNIRS VR study)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PifConfig


class CognitiveLoadScorer:
    """Binarize raw cognitive load ratings and report class distributions.

    Parameters
    ----------
    config : PifConfig
        Must contain `cl_threshold` and `label_column`.
    """

    def __init__(self, config: PifConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def binarize(self, y: np.ndarray) -> np.ndarray:
        """Apply threshold binarization to a 1-D label array.

        Parameters
        ----------
        y : np.ndarray
            Raw cognitive load scores (integer or float).

        Returns
        -------
        y_binary : np.ndarray of int
            0 = low cognitive load, 1 = high cognitive load.
        """
        T = self.config.cl_threshold
        y_binary = (y > T).astype(int)
        self._report_distribution(y_binary)
        return y_binary

    def binarize_dataframe(
        self, df: pd.DataFrame, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Binarize multiple CL columns in a DataFrame in-place.

        Parameters
        ----------
        df : pd.DataFrame
        columns : list of str, optional
            Columns to binarize. Defaults to config.cl_label_columns.

        Returns
        -------
        df : pd.DataFrame (modified copy)
        """
        df = df.copy()
        T = self.config.cl_threshold
        cols = columns or self.config.cl_label_columns
        for col in cols:
            if col in df.columns:
                df[col] = (df[col] > T).astype(int)
                print(f"[CognitiveLoadScorer] '{col}' binarized at threshold={T}")
        return df

    def per_participant_distribution(
        self,
        y: np.ndarray,
        participant_ids: np.ndarray,
    ) -> pd.DataFrame:
        """Report class distribution per participant (useful for LOSO sanity check).

        Returns
        -------
        pd.DataFrame with columns:
            Participant, Total, Class_0, Class_1, Pct_High_Load, Status
        """
        records = []
        for pid in np.unique(participant_ids):
            mask = participant_ids == pid
            y_p = y[mask]
            c0 = int(np.sum(y_p == 0))
            c1 = int(np.sum(y_p == 1))
            total = len(y_p)
            pct = c1 / total * 100 if total > 0 else 0.0

            if c1 == 0:
                status = "Only Class 0"
            elif c0 == 0:
                status = "Only Class 1"
            else:
                status = "Both classes"

            records.append(
                dict(
                    Participant=pid,
                    Total=total,
                    Class_0=c0,
                    Class_1=c1,
                    Pct_High_Load=round(pct, 1),
                    Status=status,
                )
            )

        report = pd.DataFrame(records)
        print(f"\n[CognitiveLoadScorer] Per-participant distribution:\n{report.to_string(index=False)}\n")
        return report

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _report_distribution(self, y_binary: np.ndarray) -> None:
        total = len(y_binary)
        c1 = int(np.sum(y_binary == 1))
        c0 = total - c1
        print(
            f"[CognitiveLoadScorer] Binarized at threshold={self.config.cl_threshold} | "
            f"Total={total} | Class 0 (low)={c0} | Class 1 (high)={c1} | "
            f"Balance={c1/total*100:.1f}% high-load"
        )
