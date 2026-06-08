# -*- coding: utf-8 -*-
"""
pif/feature_extractor.py
=========================
Hemodynamic and spectral feature extraction stubs.

This module is a structured stub — extend each method with your
signal-processing library of choice (MNE, nilearn, scipy, etc.).

Modalities handled:
  - fNIRS : HbO/HbR channel statistics, slope, peak amplitude
  - EEG   : Band power (delta, theta, alpha, beta, gamma), alpha/theta ratio
  - GSR   : SCR peaks, tonic level, area under curve
  - Eye   : Pupil dilation mean/std, fixation duration, saccade rate
"""

from __future__ import annotations

import numpy as np
from .config import PifConfig


class FeatureExtractor:
    """Extract modality-specific features from raw physiological signals.

    Parameters
    ----------
    config : PifConfig
        Uses config.modality to dispatch to the correct extraction method.
    """

    def __init__(self, config: PifConfig) -> None:
        self.config = config
        self._dispatch = {
            "fnirs": self.extract_fnirs,
            "eeg": self.extract_eeg,
            "gsr": self.extract_gsr,
            "eye": self.extract_eye,
        }

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """Dispatch to modality-specific extractor.

        Parameters
        ----------
        signal : np.ndarray
            Raw signal array. Shape conventions depend on modality:
              fNIRS/EEG : (n_channels, n_timepoints)
              GSR/Eye   : (n_timepoints,)

        Returns
        -------
        features : np.ndarray, shape (n_features,)
        """
        modality = self.config.modality
        if modality not in self._dispatch:
            raise ValueError(
                f"Unsupported modality '{modality}'. "
                f"Choose from: {list(self._dispatch.keys())}"
            )
        return self._dispatch[modality](signal)

    # ------------------------------------------------------------------ #
    # fNIRS features
    # ------------------------------------------------------------------ #

    def extract_fnirs(self, signal: np.ndarray) -> np.ndarray:
        """Extract hemodynamic features from fNIRS channels.

        Expected shape: (n_channels, n_timepoints)
        Features per channel: mean, std, slope, peak_amplitude
        """
        features = []
        for ch in signal:
            features.extend([
                ch.mean(),
                ch.std(),
                self._linear_slope(ch),
                ch.max() - ch.min(),
            ])
        return np.array(features, dtype=float)

    # ------------------------------------------------------------------ #
    # EEG features
    # ------------------------------------------------------------------ #

    def extract_eeg(self, signal: np.ndarray, fs: float = 256.0) -> np.ndarray:
        """Extract spectral band power features from EEG channels.

        Expected shape: (n_channels, n_timepoints)
        Features: delta, theta, alpha, beta, gamma power + alpha/theta ratio
        """
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }
        features = []
        for ch in signal:
            band_powers = {}
            for band, (lo, hi) in bands.items():
                band_powers[band] = self._band_power(ch, fs, lo, hi)
                features.append(band_powers[band])
            # Cognitive load proxy: alpha/theta ratio
            at_ratio = (
                band_powers["alpha"] / band_powers["theta"]
                if band_powers["theta"] > 0 else 0.0
            )
            features.append(at_ratio)
        return np.array(features, dtype=float)

    # ------------------------------------------------------------------ #
    # GSR features
    # ------------------------------------------------------------------ #

    def extract_gsr(self, signal: np.ndarray) -> np.ndarray:
        """Extract skin conductance features from a 1-D GSR signal."""
        peaks = self._count_peaks(signal)
        tonic = signal.mean()
        auc = np.trapz(signal)
        return np.array([tonic, peaks, auc], dtype=float)

    # ------------------------------------------------------------------ #
    # Eye-tracking features
    # ------------------------------------------------------------------ #

    def extract_eye(self, signal: np.ndarray) -> np.ndarray:
        """Extract pupillometry / fixation features from a 1-D pupil signal."""
        return np.array([
            signal.mean(),
            signal.std(),
            signal.max() - signal.min(),
            self._count_peaks(signal),
        ], dtype=float)

    # ------------------------------------------------------------------ #
    # Signal utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _linear_slope(x: np.ndarray) -> float:
        """Least-squares slope of a 1-D signal."""
        n = len(x)
        if n < 2:
            return 0.0
        t = np.arange(n, dtype=float)
        slope = np.polyfit(t, x, 1)[0]
        return float(slope)

    @staticmethod
    def _band_power(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
        """Approximate band power via FFT magnitude sum."""
        fft_vals = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
        mask = (freqs >= lo) & (freqs < hi)
        return float(fft_vals[mask].mean()) if mask.any() else 0.0

    @staticmethod
    def _count_peaks(x: np.ndarray, min_height_factor: float = 0.5) -> int:
        """Count local maxima above mean * min_height_factor."""
        threshold = x.mean() * min_height_factor
        peaks = 0
        for i in range(1, len(x) - 1):
            if x[i] > x[i - 1] and x[i] > x[i + 1] and x[i] > threshold:
                peaks += 1
        return peaks
