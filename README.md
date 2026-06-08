# 🧠 Perception-Immersion Framework

> A modular Python framework for measuring **perception, immersion, and cognitive load** in Virtual Reality (VR) environments using physiological signals and machine learning classifiers.

---

## 📖 Overview

This framework provides end-to-end tooling for VR cognitive load and immersion research, from raw signal ingestion through feature extraction, binarized load classification, and cross-validated model evaluation. It is designed to be **modality-agnostic** — supporting fNIRS, EEG, GSR, and eye-tracking — and **study-design-agnostic**, supporting both Leave-One-Subject-Out (LOSO) and k-fold cross-validation schemes.

### Inspiration & Prior Work

This framework builds on insights from:
- **`emjohann/anatomylearning`** (MIT) — fNIRS-based cognitive load classification comparing 2D vs. 3D/VR cardiac anatomy learning in medical students. Its LOSO pipeline, row-wise z-score normalization, threshold-based binarization of cognitive load scores, and multi-model evaluation loop (SVM, XGBoost, Random Forest, Logistic Regression, KNN, Naive Bayes, FCNN, CNN) form the methodological backbone of this framework.
- Established VR presence/immersion scales: IPQ (Igroup Presence Questionnaire), SUS (Slater-Usoh-Steed), and NASA-TLX for subjective cognitive load.

---

## 🗂️ Repository Structure

```
perception-immersion-framework/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── pif/                          # Core framework package
│   ├── __init__.py
│   ├── config.py                 # Global configuration dataclass
│   ├── signal_processor.py       # Raw signal ingestion & preprocessing
│   ├── feature_extractor.py      # Hemodynamic / spectral feature extraction
│   ├── cognitive_load.py         # CL scoring, thresholding, binarization
│   ├── immersion_score.py        # Immersion/presence composite scoring
│   ├── classifiers.py            # ML + DL classifier wrappers (LOSO / k-fold)
│   └── evaluator.py              # Metrics aggregation & report generation
│
├── experiments/
│   ├── run_loso.py               # LOSO experiment entry point
│   ├── run_kfold.py              # k-fold experiment entry point
│   └── configs/
│       └── default_config.yaml   # Default experiment config
│
├── tests/
│   ├── test_signal_processor.py
│   ├── test_cognitive_load.py
│   └── test_classifiers.py
│
└── docs/
    └── methodology.md            # Theoretical background & design decisions
```

---

## ⚙️ Installation

```bash
git clone https://github.com/zhub9006/perception-immersion-framework.git
cd perception-immersion-framework
pip install -r requirements.txt
```

**Python 3.9+ required.**

---

## 🚀 Quick Start

### 1. LOSO Evaluation (Leave-One-Subject-Out)

```python
from pif.config import PifConfig
from pif.signal_processor import SignalProcessor
from pif.cognitive_load import CognitiveLoadScorer
from pif.classifiers import LOSOClassifierPipeline

config = PifConfig(
    data_path="./data/Combined_Data.csv",
    label_column="CL_Total",
    cl_threshold=22,
    feature_prefix="Tx",
    participant_id_column="Participant ID",
)

processor = SignalProcessor(config)
X, y, participant_ids = processor.load_and_preprocess()

scorer = CognitiveLoadScorer(config)
y_binary = scorer.binarize(y)

pipeline = LOSOClassifierPipeline(config)
results = pipeline.run(X, y_binary, participant_ids)
pipeline.save_results(results, save_dir="./outputs/loso/")
```

### 2. k-Fold Evaluation

```python
from pif.classifiers import KFoldClassifierPipeline

pipeline = KFoldClassifierPipeline(config, n_splits=5)
results = pipeline.run(X, y_binary)
pipeline.save_results(results, save_dir="./outputs/kfold/")
```

---

## 🔬 Supported Modalities

| Modality | Signal Type | Notes |
|---|---|---|
| **fNIRS** | Hemodynamic (HbO/HbR) | Primary modality; validated in VR anatomy learning studies |
| **EEG** | Spectral band power | Alpha/Theta ratio for cognitive load proxy |
| **GSR / EDA** | Skin conductance | Arousal & stress correlate |
| **Eye-tracking** | Pupil dilation, fixation | Immersion & attention proxy |
| **Behavioral** | Response time, error rate | Task performance overlay |

---

## 🤖 Supported Classifiers

| Classifier | Type | Notes |
|---|---|---|
| SVM (RBF kernel) | Classical ML | Strong baseline for neurophysiological data |
| XGBoost | Gradient Boosting | Best overall in fNIRS benchmarks |
| Random Forest | Ensemble | Robust to feature collinearity |
| Logistic Regression | Linear | Interpretable baseline |
| K-Nearest Neighbors | Instance-based | Low-data friendly |
| Naive Bayes | Probabilistic | Fast screening |
| FCNN (MLP) | Deep Learning | BatchNorm + Dropout; 2-layer hidden |
| CNN (1D) | Deep Learning | Conv1D for temporal signal patterns |

---

## 📐 Cognitive Load Scoring

Cognitive load labels are derived from validated subjective rating scales and binarized using a configurable threshold:

- **CL_Total** — combined intrinsic + extraneous cognitive load
- **ICL_Average** — intrinsic cognitive load (task complexity)
- **ECL_Average** — extraneous cognitive load (interface/environment)
- **CL_All_Average** — mean across all CL dimensions

Binarization: `0 = low load (≤ threshold)`, `1 = high load (> threshold)`  
Default threshold: **22** (configurable in `PifConfig`).

---

## 📊 Immersion & Presence Scoring

The `immersion_score.py` module computes a composite **Perception-Immersion Index (PII)** from:
- Subjective questionnaire scores (IPQ, SUS, NASA-TLX)
- Physiological immersion proxies (pupil dilation, GSR peaks)
- Temporal engagement metrics (task dwell time, interaction rate)

---

## 🧪 Evaluation Metrics

All pipelines report per-fold and aggregate:
- **Accuracy**, **Precision**, **Recall**, **F1-Macro**
- Mean ± Std across folds
- Per-participant breakdown (LOSO)
- CSV export for downstream analysis

---

## 📚 References

1. Angkan, P. et al. — *Exploring Cognitive Load in Anatomy Education: A Study of 3D VR and 2D Learning Environments Using fNIRS* — [`emjohann/anatomylearning`](https://github.com/emjohann/anatomylearning) (MIT License)
2. Sweller, J. — *Cognitive Load Theory* (1988)
3. Slater, M. & Wilbur, S. — *A Framework for Immersive Virtual Environments (FIVE)* (1997)
4. Hart, S.G. & Staveland, L.E. — *NASA Task Load Index (NASA-TLX)* (1988)
5. Schubert, T. et al. — *The Experience of Presence: Factor Analytic Insights* (2001) — IPQ

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

*Framework scaffold generated from cross-referencing `emjohann/anatomylearning` (fNIRS/VR cognitive load, Python) and local research data at `/data/coursework/DataScience_Statistics/`. No proprietary data is included.*
