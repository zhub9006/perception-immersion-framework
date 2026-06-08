# -*- coding: utf-8 -*-
"""
pif/eeg_preprocessor.py
========================
EEG-specific preprocessing pipeline for cognitive load classification in VR.

Directly informed by reviewing:
  - Diaz265/EEG-Cognitive-State-Classifier  (preprocess.py, load_eeg.py)
  - harshitsingh4321/1DCNN-Mental-Workload-Classifier  (STEW dataset, z-score)

Pipeline steps
--------------
1. Load raw EEG from EDF file(s) via MNE
2. Pick EEG channels only (drop EOG, EMG, stim)
3. Bandpass filter: 0.5–45 Hz  (removes DC drift, line noise, HF artefacts)
4. Z-score normalization (global, across all channels and timepoints)
5. Epoch / segment into fixed-length windows
6. Return (X, y) arrays ready for EEGCognitiveLoadCNN or sklearn classifiers

Segmentation strategy (from load_eeg.py in Diaz265):
  - Non-overlapping windows of `window_size` samples
  - Windows shorter than `n_channels` are discarded
  - Labels assigned per-window (binary: 0=low, 1=high load)
"""

from __future__ import annotations

import sys
import importlib
from typing import List, Tuple, Optional

import numpy as np

# MNE is an optional dependency — only required for EDF loading.
try:
    mne = importlib.import_module("mne")
    _MNE_AVAILABLE = True
except ImportError:
    _MNE_AVAILABLE = False


class EEGPreprocessor:
    """Load, filter, normalize, and segment EEG data for CL classification.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels to retain (default 32).
        If the recording has fewer channels the available count is used.
    window_size : int
        Samples per segment window (default 128 = 1 s at 128 Hz).
    l_freq : float
        Low-frequency cutoff for bandpass filter (default 0.5 Hz).
    h_freq : float
        High-frequency cutoff for bandpass filter (default 45 Hz).
    normalize : bool
        Apply global z-score normalization after filtering (default True).

    Example
    -------
    >>> proc = EEGPreprocessor(n_channels=14, window_size=128)
    >>> X, y = proc.load_edf_files(
    ...     files=["subject01_task.edf", "subject01_rest.edf"],
    ...     labels=[1, 0],
    ... )
    >>> X.shape   # (n_windows, 14, 128)
    """

    def __init__(
        self,
        n_channels: int = 32,
        window_size: int = 128,
        l_freq: float = 0.5,
        h_freq: float = 45.0,
        normalize: bool = True,
    ) -> None:
        if not _MNE_AVAILABLE:
            raise ImportError(
                "mne is required for EEG preprocessing. "
                "Install it with: pip install mne"
            )
        self.n_channels = n_channels
        self.window_size = window_size
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.normalize = normalize

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load_edf_files(
        self,
        files: List[str],
        labels: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess a list of EDF files.

        Parameters
        ----------
        files : list of str
            Paths to EDF recordings. Each file = one condition / trial.
        labels : list of int, optional
            Per-file binary label (0=low load, 1=high load).
            If None, labels are assigned as ``file_index % 2``
            (alternating low/high — matches load_eeg.py convention).

        Returns
        -------
        X : np.ndarray, shape (n_windows, n_channels, window_size)
        y : np.ndarray, shape (n_windows,)
        """
        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        for i, filepath in enumerate(files):
            label = labels[i] if labels is not None else (i % 2)
            segments = self._process_single_file(filepath)
            X_list.extend(segments)
            y_list.extend([label] * len(segments))

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    def load_eegbci(
        self,
        subject: int = 1,
        runs: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience loader for the MNE EEG-BCI motor imagery dataset.

        Useful for smoke-testing the pipeline without custom data.
        Runs 6, 10, 14 correspond to imagined fist / feet movement.

        Parameters
        ----------
        subject : int
            Subject index (1-based).
        runs : list of int, optional
            BCI run indices. Default [6, 10].

        Returns
        -------
        X : np.ndarray, shape (n_windows, n_channels, window_size)
        y : np.ndarray, shape (n_windows,)
        """
        if runs is None:
            runs = [6, 10]
        files = mne.datasets.eegbci.load_data(subject, runs)
        return self.load_edf_files(files)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _process_single_file(self, filepath: str) -> List[np.ndarray]:
        """Load one EDF file → filter → normalize → segment → return windows."""
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        raw.pick_types(eeg=True)
        raw.filter(self.l_freq, self.h_freq, verbose=False)

        data = raw.get_data()  # shape: (total_channels, n_timepoints)

        # Truncate or pad channel dimension
        n_ch = min(data.shape[0], self.n_channels)
        data = data[:n_ch, :]

        # Global z-score normalization (mirrors Diaz265/preprocess.py)
        if self.normalize:
            mean = data.mean()
            std = data.std() + 1e-8
            data = (data - mean) / std

        # Segment into non-overlapping windows
        segments: List[np.ndarray] = []
        n_timepoints = data.shape[1]
        for start in range(0, n_timepoints - self.window_size, self.window_size):
            window = data[:, start: start + self.window_size]
            if window.shape[0] >= n_ch:
                segments.append(window)

        return segments
