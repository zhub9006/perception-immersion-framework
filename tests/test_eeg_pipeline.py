# -*- coding: utf-8 -*-
"""
tests/test_eeg_pipeline.py
===========================
Unit and smoke tests for the EEG preprocessing and CNN modules.

Run with:  pytest tests/test_eeg_pipeline.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pif.eeg_cnn import EEGCognitiveLoadCNN, evaluate, train_one_epoch
from pif.cognitive_load import CognitiveLoadScorer
from pif.config import PifConfig
from pif.immersion_score import ImmersionScorer


# ======================================================================== #
# EEGCognitiveLoadCNN
# ======================================================================== #

class TestEEGCognitiveLoadCNN:

    def test_output_shape_default(self):
        """Forward pass with default 32-channel, 128-timepoint input."""
        model = EEGCognitiveLoadCNN(n_channels=32, n_classes=2)
        x = torch.randn(8, 32, 128)
        out = model(x)
        assert out.shape == (8, 2), f"Expected (8, 2), got {out.shape}"

    def test_output_shape_emotiv(self):
        """Forward pass with 14-channel Emotiv headset layout."""
        model = EEGCognitiveLoadCNN(n_channels=14, n_classes=2)
        x = torch.randn(4, 14, 128)
        out = model(x)
        assert out.shape == (4, 2)

    def test_variable_window_length(self):
        """AdaptiveAvgPool1d should handle different window sizes."""
        model = EEGCognitiveLoadCNN(n_channels=32, n_classes=2)
        for wlen in [64, 128, 256, 512]:
            x = torch.randn(2, 32, wlen)
            out = model(x)
            assert out.shape == (2, 2), f"Failed for window_size={wlen}"

    def test_no_nan_in_output(self):
        """Output should not contain NaN values."""
        model = EEGCognitiveLoadCNN(n_channels=32)
        x = torch.randn(16, 32, 128)
        out = model(x)
        assert not torch.isnan(out).any(), "NaN detected in model output"

    def test_gradient_flows(self):
        """All parameters should receive gradients after backward pass."""
        model = EEGCognitiveLoadCNN(n_channels=32)
        x = torch.randn(4, 32, 128)
        y = torch.randint(0, 2, (4,))
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(model(x), y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ======================================================================== #
# CognitiveLoadScorer
# ======================================================================== #

class TestCognitiveLoadScorer:

    def setup_method(self):
        self.config = PifConfig(cl_threshold=22)
        self.scorer = CognitiveLoadScorer(self.config)

    def test_binarize_below_threshold(self):
        y = np.array([10, 15, 20, 22])
        result = self.scorer.binarize(y)
        assert (result == 0).all(), "All values ≤ 22 should be class 0"

    def test_binarize_above_threshold(self):
        y = np.array([23, 25, 30])
        result = self.scorer.binarize(y)
        assert (result == 1).all(), "All values > 22 should be class 1"

    def test_binarize_mixed(self):
        y = np.array([20, 22, 23, 30])
        result = self.scorer.binarize(y)
        np.testing.assert_array_equal(result, [0, 0, 1, 1])

    def test_output_dtype_is_int(self):
        y = np.array([18.5, 22.0, 25.3])
        result = self.scorer.binarize(y)
        assert result.dtype in (np.int32, np.int64, int)

    def test_class_distribution_returns_dict(self):
        y = np.array([10, 20, 25, 30])
        dist = self.scorer.class_distribution(y)
        assert isinstance(dist, dict)
        assert set(dist.keys()) == {0, 1}


# ======================================================================== #
# ImmersionScorer
# ======================================================================== #

class TestImmersionScorer:

    def setup_method(self):
        self.config = PifConfig()
        self.scorer = ImmersionScorer(self.config)

    def test_pii_range(self):
        """PII values must be in [0, 1]."""
        q = np.array([[4.0, 3.5, 4.2], [2.0, 2.5, 3.0]])
        p = np.array([[0.7, 10.0], [0.3, 4.0]])
        b = np.array([[50.0, 8.0], [20.0, 3.0]])
        pii = self.scorer.compute_pii(q, p, b)
        assert ((pii >= 0) & (pii <= 1)).all(), f"PII out of [0,1]: {pii}"

    def test_pii_shape(self):
        """Output length must equal number of subjects."""
        n_subjects = 5
        q = np.random.rand(n_subjects, 3) * 5
        p = np.random.rand(n_subjects, 2)
        b = np.random.rand(n_subjects, 2) * 60
        pii = self.scorer.compute_pii(q, p, b)
        assert pii.shape == (n_subjects,)

    def test_higher_scores_give_higher_pii(self):
        """A subject with uniformly higher scores should have higher PII."""
        q_high = np.array([[5.0, 5.0, 5.0]])
        q_low  = np.array([[1.0, 1.0, 1.0]])
        p = np.array([[0.5, 8.0]])
        b = np.array([[40.0, 6.0]])
        pii_high = self.scorer.compute_pii(q_high, p, b)[0]
        pii_low  = self.scorer.compute_pii(q_low, p, b)[0]
        assert pii_high >= pii_low, "Higher questionnaire scores should yield higher PII"
