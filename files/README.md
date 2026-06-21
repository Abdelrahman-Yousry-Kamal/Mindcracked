# EEG Motor-Imagery Models & Training Pipelines

This folder contains the **model architectures and training scripts** used as the surrogate / black-box targets for the [BCI IP Fingerprinting & Verification framework](../ADV-TRA_EEG_BCI). It implements a complete benchmark suite of motor-imagery EEG classifiers — from the project's own Physics-Informed Neural Network (PINN) to standard baselines like EEGNet and EEG-Conformer — all trained and evaluated on the **BCI Competition IV-2a** dataset.

These models exist to stress-test the fingerprinting pipeline: the PINN is the protected source model, while the other architectures (EEGNet, DeepConvNet, ShallowConvNet, FBCNet, EEGTCNet, EEG-Conformer) simulate independently-trained or distilled "black-box" derivatives that a verifier must still be able to recognize as related — or correctly reject as unrelated.

---

## 📂 Folder Structure

```text
files/
├── main.py                    # End-to-end EEGNet training + official holdout evaluation
├── PINN_TRAIN.py               # Standalone training script for the Physics-Informed Neural Network
├── EEGConformer_TRAIN.py       # Multi-subject (9-subject) EEG-Conformer training + benchmarking loop
├── preprocessing.py            # Causal EEG preprocessing: EOG regression, bandpass filtering, windowing
├── requirements.txt
├── models/
│   ├── PINN.py                 # NM_PINN — Wilson-Cowan physics-informed classifier (the protected model)
│   ├── EEGNet.py                # Compact CNN baseline (Lawhern et al., 2018)
│   ├── deepCNN.py               # DeepConvNet baseline
│   ├── shallowCNN.py            # ShallowConvNet baseline
│   ├── FBCNet.py                 # Filter-Bank Convolutional Network
│   ├── TCNet.py                  # EEG-TCNet (temporal convolutional network)
│   └── EEGConformer.py           # Wrapper around braindecode's EEG-Conformer (transformer-based)
└── layers/
    ├── layers.py                 # Max-norm constrained Linear/Conv2d layers (Keras-style constraints)
    └── parametrization.py        # MaxNorm weight parametrizations used by the constrained layers
```

---

## 🧠 Task & Dataset

All models classify 4-class **motor imagery** from EEG:

| Class | Label |
|---|---|
| 0 | Left Hand |
| 1 | Right Hand |
| 2 | Both Feet |
| 3 | Tongue |

Trained and evaluated on the **BCI Competition IV-2a** dataset (22 EEG channels + 3 EOG channels, 250 Hz, 9 subjects, session `T` for training / session `E` for evaluation).

---

## ⚙️ Preprocessing Pipeline (`preprocessing.py`)

`BCICausalPreprocessor` implements a strictly **causal**, leak-proof signal pipeline:

1. **NaN handling** — replaces invalid samples from raw GDF logs with zeros.
2. **EOG artifact removal** — fits a linear regression of the 3 EOG channels against the 22 EEG channels on training data only (`fit_eog_regression`), then subtracts the eye-movement contribution (`apply_eog_regression`).
3. **Causal bandpass filtering** — a 4th-order Butterworth filter (default 8–30 Hz, the mu/beta motor-imagery band) applied with `scipy.signal.lfilter`, i.e. no forward-backward (zero-phase) filtering, so no future samples leak into the past.
4. **Causal windowing** (`generate_causal_windows`) — slices 2-second windows with a 200ms stride between 0.5s–3.5s after each cue, using only `769/770/771/772` (training) or `783` (evaluation) event markers from the GDF annotations.
5. **Trial-level evaluation** (`evaluate` / `visualize`) — aggregates window-level logits back into trial-level predictions via mean-logit pooling, reconstructs the true trial order from the relative timeline clock, and renders a confusion matrix.

This same preprocessor is reused identically across every training script to guarantee a fair, leakage-free comparison between architectures.

---

## 🏗️ Model Zoo (`models/`)

| Model | File | Description |
|---|---|---|
| **NM_PINN** | `PINN.py` | The project's protected source architecture. An EEGNet-style temporal + depthwise spatial front-end feeds a **Wilson-Cowan neural-mass model** physics branch, splitting depthwise filters into Excitatory/Inhibitory channel pairs. Outputs classification logits plus two regularization losses: a Wilson-Cowan ODE residual loss (`loss_wc`) enforcing biologically plausible E/I dynamics, and an orthogonality loss (`loss_ortho`) keeping E and I spatial filters decorrelated. |
| **EEGNet** | `EEGNet.py` | PyTorch port of Lawhern et al.'s compact CNN (temporal conv → depthwise spatial conv → separable conv), with max-norm constrained layers mirroring the original Keras implementation. |
| **DCNN (DeepConvNet)** | `deepCNN.py` | A deeper 4-block CNN baseline; notes in the file flag it as prone to overfitting past ~300 epochs. |
| **SCNN (ShallowConvNet)** | `shallowCNN.py` | A shallow square/log-power CNN baseline; notes flag it as prone to fully overfitting the training set. |
| **FBCNet** | `FBCNet.py` | Filter-Bank CNN: splits the signal into 8 frequency sub-bands (4–36 Hz) via SciPy SOS Butterworth filters, applies per-band spatial convolutions, then temporal variance pooling. Includes a two-stage train/early-stopping procedure per the original paper. Marked **"Not working"** in this codebase. |
| **EEGTCNet** | `TCNet.py` | EEGNet-style front-end followed by a stack of dilated causal residual TCN blocks; the file recommends ≥750 training epochs for convergence. |
| **EEGConformer** | `EEGConformer.py` | Thin wrapper around `braindecode.models.EEGConformer`, a transformer-based architecture, used as a structurally very different black-box target. |

All non-conformer models expose a static `training(...)` helper with a standard AdamW + cross-entropy + gradient-clipping training loop, so any architecture can be dropped into the same pipeline.

### Constrained layers (`layers/`)

`layers.py` and `parametrization.py` reimplement Keras-style `max_norm` weight constraints in PyTorch using `torch.nn.utils.parametrize`:
- `MaxNormLinear` / `MaxNorm` — used by EEGNet's final dense layer.
- `LinearWithConstraint` / `Conv2dWithConstraint` / `MaxNormParametrize` — general-purpose max-norm constrained Linear and Conv2d layers used across DeepConvNet and ShallowConvNet.

---

## 🚀 Training Scripts

### `main.py` — EEGNet single-subject pipeline
Loads one subject's GDF training/evaluation files (Kaggle dataset paths), preprocesses with `BCICausalPreprocessor`, trains an `EEGNet` instance for 300 epochs (AdamW, label-smoothed cross-entropy, gradient clipping), then runs the **official holdout evaluation**: reconstructs trial-level labels from the relative timeline clock and reports trial-level accuracy with a confusion matrix.

### `PINN_TRAIN.py` — Physics-Informed Neural Network training
Interactive script (prompts for subject 1–9). Trains `NM_PINN` for 350 epochs with:
- **Differential learning rates** — base layers at `1e-3`, physics parameters (`tau_E`, `tau_I`, `w_EE`, `w_EI`, `w_IE`, `w_II`, `P`, `Q`) at `1e-4`.
- **Warmup-scheduled physics loss weight** — the Wilson-Cowan ODE residual term ramps from 0 to `0.002` after epoch 10.
- A fixed orthogonality loss weight of `0.01`.

Saves normalization statistics (`mean`, `std`, `eog_weights`) alongside the trained checkpoint so downstream verification/attack scripts can reproduce identical preprocessing.

### `EEGConformer_TRAIN.py` — Multi-subject benchmark loop
Trains a **fresh EEG-Conformer per subject** (all 9 subjects, 200 epochs each, AdamW + cosine annealing), evaluates each on its official holdout session, and aggregates a global confusion matrix and mean accuracy across all subjects — used to benchmark the transformer-based architecture against the project's PINN and CNN baselines.

---

## 🛠️ Setup

```bash
pip install -r requirements.txt
```

Dependencies: `mne`, `numpy`, `scipy`, `seaborn`, `scikit-learn`, `matplotlib`, `torch`, `torchaudio`, `torchvision` (plus `braindecode`, required by `EEGConformer.py` but not pinned in `requirements.txt`).

Training scripts expect the BCI Competition IV-2a `.gdf`/`.mat` files at Kaggle-style input paths (e.g. `/kaggle/input/datasets/abdelrahmanyousryyu/bci-comp2a/A0{subject}T.gdf`); update these paths if running outside Kaggle.

```bash
python main.py                  # EEGNet, subject 1 (edit `subject` in-file to change)
python PINN_TRAIN.py            # prompts for subject number, trains NM_PINN
python EEGConformer_TRAIN.py    # loops over all 9 subjects automatically
```

---

## 🔗 Related

This folder is consumed by the [`ADV-TRA_EEG_BCI`](../ADV-TRA_EEG_BCI) fingerprinting framework, which uses these architectures as **Black-Box Models** during adversarial verification — auditing whether a fine-tuned or structurally mismatched checkpoint (e.g. a stolen PINN retrained as an EEGNet-style network) still carries traceable decision-boundary fingerprints of the original protected model. The `Attacking_Phase` folder builds on these baselines to implement concrete model-stealing and fingerprint-evasion attacks.
