# -*- coding: utf-8 -*-
"""
pif/immersion_score.py
======================
Composite Perception-Immersion Index (PII) computation.

The PII aggregates three evidence streams:
  1. Questionnaire scores  (IPQ, SUS, NASA-TLX) — subjective
  2. Physiological proxies (pupil dilation, GSR peaks) — objective
  3. Behavioral metrics    (dwell time, interaction rate) — task-level

Each stream is normalized to [0, 1] and combined via configurable weights
(default: 50% / 30% / 20%).  The final PII ∈ [0, 1] where higher values
indicate greater perceived immersion / presence.

Usage
-----
    from pif.config import PifConfig
    from pif.immersion_score import ImmersionScorer
    import numpy as np

    config = PifConfig()
    scorer = ImmersionScorer(config)

    # Questionnaire data (n_subjects × n_items)
    q_scores = np.array([[4.2, 3.8, 4.5], [2.1, 2.9, 3.0]])

    # Physiological features per subject: [mean_pupil_dilation, gsr_peak_count]
    physio = np.array([[0.65, 12], [0.40, 6]])

    # Behavioral features per subject: [mean_dwell_time_s, interaction_rate_per_min]
    behavioral = np.array([[45.0, 8.2], [30.0, 4.1]])

    pii = scorer.compute_pii(q_scores, physio, behavioral)
    print(pii)  # array([0.78, 0.45])
"""

from __future__ import annotations

import numpy as np

from .config import PifConfig


class ImmersionScorer:
    """Compute the composite Perception-Immersion Index (PII).

    Parameters
    ----------
    config : PifConfig
        Weights for each evidence stream are read from:
            config.pii_weight_questionnaire
            config.pii_weight_physiological
            config.pii_weight_behavioral
    """

    def __init__(self, config: PifConfig) -> None:
        self.config = config
        self._validate_weights()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def compute_pii(
        self,
        questionnaire: np.ndarray,
        physiological: np.ndarray,
        behavioral: np.ndarray,
    ) -> np.ndarray:
        """Compute per-subject PII scores.

        Parameters
        ----------
        questionnaire : np.ndarray, shape (n_subjects, n_q_items)
            Raw questionnaire responses (e.g., Likert 1–7).
        physiological : np.ndarray, shape (n_subjects, n_physio_features)
            Physiological signal features (e.g., pupil dilation, GSR peaks).
        behavioral : np.ndarray, shape (n_subjects, n_behav_features)
            Behavioral metrics (e.g., dwell time, interaction rate).

        Returns
        -------
        pii : np.ndarray, shape (n_subjects,)
            Composite immersion index in [0, 1] per subject.
        """
        q_norm = self._normalize_stream(questionnaire.mean(axis=1))
        p_norm = self._normalize_stream(physiological.mean(axis=1))
        b_norm = self._normalize_stream(behavioral.mean(axis=1))

        pii = (
            self.config.pii_weight_questionnaire * q_norm
            + self.config.pii_weight_physiological * p_norm
            + self.config.pii_weight_behavioral * b_norm
        )

        pii = np.clip(pii, 0.0, 1.0)
        print(
            f"[ImmersionScorer] PII computed for {len(pii)} subjects | "
            f"Mean={pii.mean():.3f} ± {pii.std():.3f}"
        )
        return pii

    def compute_pii_from_dict(self, data: dict) -> np.ndarray:
        """Convenience wrapper accepting a dict with keys
        'questionnaire', 'physiological', 'behavioral'.
        """
        return self.compute_pii(
            data["questionnaire"],
            data["physiological"],
            data["behavioral"],
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_stream(x: np.ndarray) -> np.ndarray:
        """Min-max normalize a 1-D array to [0, 1]."""
        x_min, x_max = x.min(), x.max()
        if x_max == x_min:
            return np.zeros_like(x, dtype=float)
        return (x - x_min) / (x_max - x_min)

    def _validate_weights(self) -> None:
        total = (
            self.config.pii_weight_questionnaire
            + self.config.pii_weight_physiological
            + self.config.pii_weight_behavioral
        )
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(
                f"PII weights must sum to 1.0 (got {total:.4f}). "
                "Check pii_weight_* fields in PifConfig."
            )
