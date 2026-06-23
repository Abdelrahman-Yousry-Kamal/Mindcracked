# NM-PINN: Neural Mass Physics-Informed Neural Network for Motor Imagery EEG Decoding

**Intellectual Property Protection for Brain-Computer Interface Models**

Mindcracked is a framework for protecting and verifying ownership of EEG-based motor-imagery classifiers. It combines a custom **Physics-Informed Neural Network (PINN)** for BCI decoding with a non-intrusive **fingerprinting and verification system**, then stress-tests that protection against realistic model-theft scenarios — fine-tuning, distillation, pruning, and cross-architecture attacks.

The core idea: a model owner generates a small set of **adversarial trajectory fingerprints** — boundary-crossing EEG perturbations baked into the trained model's decision geometry. These act as a "DNA watermark." Even if an attacker steals the model and retrains it on different data, prunes it, or distills it into an entirely different architecture, the verifier can still detect shared lineage by checking how the suspect model responds to these fingerprint trajectories.

---

## Repository Structure

```text
Mindcracked/
├── ADV-TRA_EEG_BCI/      # Core IP fingerprinting & verification pipeline
├── Attacking_Phase/       # Implementations of model-theft / fingerprint-evasion attacks
├── files/                 # EEG model zoo + training pipelines (PINN, EEGNet, EEGConformer, etc.)
└── README.md
```

| Folder | Purpose |
|---|---|
| **`ADV-TRA_EEG_BCI`** | End-to-end pipeline: data setup → fingerprint generation → baseline evaluation → adversarial fine-tuning simulation → batch verification/audit against suspect checkpoints. |
| **`files`** | The underlying EEG motor-imagery model architectures (the protected PINN plus black-box comparison baselines) and their training/preprocessing code. Used both to train the source model and to supply "black-box" attack/comparison targets for the verifier. |
| **`Attacking_Phase`** | Implementations of IP-theft attacks against the fingerprinted model — used to validate how robust the verification framework actually is. |

---

## The Problem

Motor-imagery BCI models (e.g. classifying *Left Hand / Right Hand / Both Feet / Tongue* imagery from EEG) are expensive to train and tune — they require carefully cleaned signal data, subject-specific calibration, and physics-aware architectures. Once a trained model is exposed (via an API, a leaked checkpoint, or a research release), it can be:

- **Fine-tuned** on new data to disguise its origin,
- **Distilled** into a different, cheaper architecture,
- **Pruned** to mask its parameter signature,

while still functionally relying on the stolen decision boundaries. Mindcracked asks: *can we still prove this model is a derivative of ours, even after all of that?*

---

## How It Fits Together

```text
            ┌───────────────┐
            │   files/      │  Train PINN (protected model) +
            │  model zoo    │  black-box comparison architectures
            └──────┬────────┘
                   │ source_model.pth
                   ▼
   ┌───────────────────────────────┐
   │   ADV-TRA_EEG_BCI/             │  Generate adversarial trajectory
   │  fingerprinting & verification │  fingerprints from the trained PINN
   └───────────────┬────────────────┘
                   │ fingerprint_path/
                   ▼
   ┌───────────────────────────────┐
   │   Attacking_Phase/              │  Attempt to steal / launder the
   │  IP-theft attack implementations│  model and erase its watermark
   └───────────────┬────────────────┘
                   │ attacks/*.pth
                   ▼
   back into ADV-TRA_EEG_BCI's batch verifier → IP Alarm if fingerprint match persists
```

1. **Train** the source PINN and comparison architectures using the code in `files/`.
2. **Fingerprint** the trained PINN using `ADV-TRA_EEG_BCI`, generating boundary-crossing trajectory keys unique to its decision geometry.
3. **Attack** the model using the techniques in `Attacking_Phase/` (fine-tuning on new subject data, pruning, cross-architecture distillation, etc.), simulating a realistic theft scenario.
4. **Verify**: run the resulting checkpoints — whether matched-architecture derivatives or structurally mismatched black-box models — back through the `ADV-TRA_EEG_BCI` auditor, which flags ownership if fingerprint deviation stays below threshold (Mutation Deviation < 45%).

---

## Component Details

### 1. `files/` — EEG Model Zoo & Training Pipelines

Implements and trains the architectures used throughout the project, all targeting 4-class motor-imagery classification on the **BCI Competition IV-2a** dataset (22 EEG channels, 250 Hz, 9 subjects).

- **`preprocessing.py`** — `BCICausalPreprocessor`: causal (no future leakage) signal pipeline — EOG artifact regression, 4th-order Butterworth bandpass filtering (8–30 Hz mu/beta band), causal windowing around motor-imagery cues, and trial-level evaluation via mean-logit pooling.
- **`models/`**:
  - **`PINN.py` (`NM_PINN`)** — the protected source architecture. An EEGNet-style temporal + depthwise spatial front-end feeds a **Wilson-Cowan neural-mass model** physics branch (Excitatory/Inhibitory channel pairs), trained with an ODE-residual physics loss and a spatial orthogonality loss alongside standard cross-entropy.
  - **`EEGNet.py`, `deepCNN.py`, `shallowCNN.py`, `FBCNet.py`, `TCNet.py`, `EEGConformer.py`** — standard/state-of-the-art BCI baselines (compact CNN, DeepConvNet, ShallowConvNet, filter-bank CNN, temporal-convolutional network, transformer-based Conformer) used as black-box comparison and attack targets.
- **`layers/`** — Keras-style max-norm weight constraints reimplemented in PyTorch via `torch.nn.utils.parametrize`.
- **Training scripts**: `main.py` (single-subject EEGNet + holdout eval), `PINN_TRAIN.py` (interactive PINN training with differential learning rates and warmup-scheduled physics loss), `EEGConformer_TRAIN.py` (full 9-subject Conformer benchmark loop).

### 2. `ADV-TRA_EEG_BCI/` — Fingerprinting & Verification Pipeline

A 7-step pipeline (see its own README for full command-by-command detail):

1. **Environment setup** — isolated directory tree for data, models, attacks, and fingerprints.
2. **Dataset procurement** — downloads the authentic BCI Competition IV-2a dataset via `kagglehub`.
3. **Data partitioning** — formats raw GDF signals into PyTorch tensor blocks.
4. **Fingerprint generation** — the core routine: optimizes 10 trajectory sequences that introduce micro-volt perturbations to baseline EEG slices until they smoothly cross a decision boundary, saving these as immutable watermark keys.
5. **Baseline evaluation** — establishes raw multi-class accuracy on held-out subject sessions.
6. **Adversarial fine-tuning simulation** — retrains the stolen model on a different subject's session to simulate an IP-theft/laundering attempt.
7. **Batch verification & audit** — scans a folder of suspect checkpoints (`attacks/` or `Black_Box_Models/`), auto-detecting architecture mismatches and routing into an architecture-agnostic verification path (`torch.nn.functional.linear` boundary mapping) when needed. Outputs a Mutation Deviation score; values under 45% raise an IP ownership alarm regardless of structural disguise.

### 3. `Attacking_Phase/`

Implements the adversarial side of the project: attacks against the fingerprinted PINN intended to test whether an adversary can erase, evade, or launder the watermark — for example by fine-tuning on new data, pruning weights, or distilling into one of the alternative architectures from `files/models/`. These attack outputs are exactly what `ADV-TRA_EEG_BCI`'s verifier is built to catch.

> This section is based on a summary description rather than a full code review of the folder. If you'd like a detailed breakdown of the specific attacks implemented here (algorithms, parameters, scripts), share the folder contents and this section can be expanded.

---

## Setup

```bash
pip install -r files/requirements.txt
```

Core dependencies: `mne`, `numpy`, `scipy`, `seaborn`, `scikit-learn`, `matplotlib`, `torch`, `torchaudio`, `torchvision`, plus `kagglehub` (dataset download) and `braindecode` (required by `EEGConformer.py`, not currently pinned in `requirements.txt`).

Most scripts assume Kaggle/Colab-style paths (`/kaggle/input/...`, `/content/...`) for the BCI Competition IV-2a `.gdf`/`.mat` files — update these paths if running locally.

---

## Project Context

This is a graduation project applying physics-informed deep learning and adversarial watermarking techniques to the problem of intellectual-property protection for brain-computer interface models — an emerging concern as BCI decoders move from research settings into deployed products and APIs.
