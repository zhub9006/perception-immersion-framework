# -*- coding: utf-8 -*-
"""
pif/__init__.py
===============
Perception-Immersion Framework — public API surface.
"""

from .config import PifConfig
from .signal_processor import SignalProcessor
from .cognitive_load import CognitiveLoadScorer
from .immersion_score import ImmersionScorer
from .classifiers import LOSOClassifierPipeline, KFoldClassifierPipeline
from .evaluator import Evaluator

__version__ = "0.1.0"
__all__ = [
    "PifConfig",
    "SignalProcessor",
    "CognitiveLoadScorer",
    "ImmersionScorer",
    "LOSOClassifierPipeline",
    "KFoldClassifierPipeline",
    "Evaluator",
]
