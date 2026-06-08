# Methodology & Theoretical Background

> **perception-immersion-framework** — design rationale, signal choices, and literature anchors.

---

## 1. Cognitive Load Theory (Sweller, 1988)

Cognitive Load Theory partitions mental effort into three components:

| Component | Definition | Measurement proxy |
|---|---|---|
| **Intrinsic CL (ICL)** | Inherent complexity of the task/content | Subjective rating scales (NASA-TLX, CLQ) |
| **Extraneous CL (ECL)** | Load imposed by poor interface/environment design | Questionnaire + error rate |
| **Germane CL (GCL)** | Effort directed at schema formation | Learning performance metrics |

In VR contexts, ECL is particularly sensitive to rendering fidelity, latency, and navigation complexity — making it a primary target for immersion research.

---

## 2. Immersion vs. Presence

These terms are often conflated; this framework treats them as distinct but related:

- **Immersion** (Slater & Wilbur, 1997) — an *objective*, technology-driven property of the VR system (field of view, frame rate, tracking latency, stereoscopy). Immersion can be measured from device specs.
- **Presence** — a *subjective* psychological state ("being there") that immersion enables but does not guarantee. Measured via validated questionnaires.

### Validated Presence Scales Supported

| Scale | Items | What it measures |
|---|---|---|
| **IPQ** (Igroup Presence Questionnaire) | 14 items | Spatial presence, involvement, experienced realism |
| **SUS** (Slater-Usoh-Steed) | 6 items | Overall sense of "being there" |
| **NASA-TLX** | 6 subscales | Mental demand, physical demand, temporal demand, performance, effort, frustration |

---

## 3. Physiological Signals for Cognitive Load

### 3.1 EEG (Electroencephalography)

EEG is the most established neuroimaging modality for real-time CL assessment.

**Key frequency bands and CL relevance:**

| Band | Frequency | CL relevance |
|---|---|---|
| Delta (δ) | 0.5–4 Hz | Drowsiness, fatigue |
| Theta (θ) | 4–8 Hz | ↑ with working memory load (frontal midline) |
| Alpha (α) | 8–13 Hz | ↓ with increasing cognitive demand |
| Beta (β) | 13–30 Hz | Active thinking, motor preparation |
| Gamma (γ) | 30–45 Hz | High-level cognition, feature binding |

**Primary CL proxy:** Alpha/Theta power ratio (↓ ratio = higher load).

**Pipeline (implemented in `EEGPreprocessor`):**
1. Bandpass 0.5–45 Hz (removes DC drift and HF artefacts)
2. Z-score normalization (global or row-wise per sample)
3. Windowing into 128-sample (1 s @ 128 Hz) non-overlapping epochs
4. 1D CNN classification via `EEGCognitiveLoadCNN`

**Dataset benchmark:** STEW (Sustained Attention to Response Task EEG Workload) — 14 Emotiv channels, ratings 4–9, binary split at 6.5.

### 3.2 fNIRS (functional Near-Infrared Spectroscopy)

fNIRS measures hemodynamic responses (HbO/HbR) in the prefrontal cortex — the primary CL-sensitive region.

**Advantages over EEG for VR:**
- Tolerant of movement artefacts (relevant in 6DOF VR)
- No electrode gel required
- Compatible with HMD form factors (Oculus/VIVE)

**Pipeline (implemented in `SignalProcessor`):**
- Row-wise z-score normalization per sample (removes inter-subject amplitude differences)
- Feature extraction: mean HbO/HbR, slope, peak amplitude per channel
- LOSO cross-validation to account for subject-level variability

**Reference study:** Angkan et al. — *fNIRS-based CL in 3D VR vs. 2D anatomy learning* (`emjohann/anatomylearning`).

### 3.3 GSR / EDA (Galvanic Skin Response)

GSR measures sympathetic arousal (stress, cognitive effort, emotional engagement).

- **Tonic level** (SCL): slow-changing baseline → general arousal
- **Phasic peaks** (SCR): fast responses to stimuli → event-locked CL spikes

### 3.4 Eye-Tracking

- **Pupil dilation**: strong correlate of cognitive load (task-evoked pupillary response, TEPR)
- **Fixation duration**: longer fixations → higher local processing demand
- **Saccade rate**: fewer saccades → higher focused attention

---

## 4. Classification Architecture

### 4.1 Classical ML Baseline

Following `emjohann/anatomylearning`, the framework evaluates 6 classical models as baselines:
SVM (RBF), XGBoost, Random Forest, Logistic Regression, KNN, Naive Bayes.

XGBoost consistently outperforms in fNIRS benchmarks; SVM is the strongest EEG baseline.

### 4.2 Deep Learning

**FCNN (Fully Connected NN):**
- 2 hidden layers with BatchNorm + Dropout
- Input: flattened feature vector
- Suitable for tabular fNIRS features

**1D CNN (`EEGCognitiveLoadCNN`):**
- 3 Conv1d blocks (64 → 128 → 256 filters, kernels 7 → 5 → 3)
- AdaptiveAvgPool1d(1) for variable-length input
- Classifier: Linear(256→128) → ReLU → Dropout(0.4) → Linear(128→2)
- Architecture informed by Diaz265/EEG-Cognitive-State-Classifier

---

## 5. Cross-Validation Strategy

### Leave-One-Subject-Out (LOSO)

LOSO is the gold standard for neurophysiological ML experiments:
- Each fold trains on N-1 subjects, tests on the held-out subject
- Directly measures *cross-subject generalizability* — critical for VR deployment
- Recommended when N < 30 subjects

### k-Fold (Stratified)

Used when subject counts are large or for ablation studies:
- Default: 5-fold stratified
- Faster than LOSO; risks data leakage if subjects contribute to multiple folds

---

## 6. Perception-Immersion Index (PII)

The PII is a composite [0, 1] score fusing three evidence streams:

```
PII = w_q × Q_norm + w_p × P_norm + w_b × B_norm
```

Where:
- **Q_norm**: normalized questionnaire score (IPQ/SUS mean, scaled to [0,1])
- **P_norm**: physiological immersion proxy (pupil dilation + GSR peaks, scaled)
- **B_norm**: behavioral engagement metric (dwell time + interaction rate, scaled)
- Default weights: w_q=0.50, w_p=0.30, w_b=0.20

Implemented in `pif/immersion_score.py :: ImmersionScorer.compute_pii()`.

---

## 7. Key References

1. Sweller, J. (1988). *Cognitive load during problem solving: Effects on learning.* Cognitive Science, 12(2), 257–285.
2. Slater, M. & Wilbur, S. (1997). *A framework for immersive virtual environments (FIVE).* Presence, 6(6), 603–616.
3. Schubert, T., Friedmann, F., & Regenbrecht, H. (2001). *The experience of presence: Factor analytic insights.* Presence, 10(3), 266–281.
4. Hart, S.G. & Staveland, L.E. (1988). *Development of NASA-TLX.* Advances in Psychology, 52, 139–183.
5. Lim, W.L. et al. (2018). *STEW: Simultaneous task EEG workload dataset.* IEEE TNSRE, 26(12), 2315–2323.
6. Angkan, P. et al. (2024). *Exploring Cognitive Load in Anatomy Education: A Study of 3D VR and 2D Learning Environments Using fNIRS.* [`emjohann/anatomylearning`](https://github.com/emjohann/anatomylearning)
7. Diaz265. (2024). *EEG Cognitive State Classifier* (PyTorch). [`Diaz265/EEG-Cognitive-State-Classifier`](https://github.com/Diaz265/EEG-Cognitive-State-Classifier)
8. harshitsingh4321. (2024). *1DCNN Mental Workload Classifier* (STEW, TF/Keras). [`harshitsingh4321/1DCNN-Mental-Workload-Classifier`](https://github.com/harshitsingh4321/1DCNN-Mental-Workload-Classifier)
