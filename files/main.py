import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import scipy.io as sio
from models.EEGNet import EEGNet
from preprocessing import BCICausalPreprocessor

architecture = EEGNet
subject = 1
# 0. Instantiate the Preprocessor FIRST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Initializing Preprocessor...")
preprocessor = BCICausalPreprocessor(lowcut=8.0, highcut=30.0)

# 1. PROCESS TRAINING DATA
print("\nProcessing Training Data...")
train_file = f'/kaggle/input/datasets/abdelrahmanyousryyu/bci-comp2a/A0{subject}T.gdf'
train_eeg, train_events, train_dict = preprocessor.process_file(train_file, is_training=True)


X_train, y_train, _ = BCICausalPreprocessor.generate_causal_windows(
    train_eeg, train_events, train_dict, window_size_sec=2.0, is_training=True
)


train_mean = np.mean(X_train, axis=-1, keepdims=True)
train_std = np.std(X_train, axis=-1, keepdims=True) + 1e-8
X_train_norm = (X_train - train_mean) / train_std

X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)


batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)


model = architecture(nb_classes=4,
        Chans=22,
        Samples=500,
        dropoutRate=0.5,
        kernLength=22,
        F1=8,
        D=4,
        F2=32,
        norm_rate=0.25,
        dropoutType="Dropout",
        pk1=4,
        pk2=8,)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)

architecture.training(model, criterion, optimizer, 300, 0.001, train_loader, device)

print(f"Running Official Holdout Evaluation (Session T -> Session E) on device: {device}\n")
print("Loading official evaluation true labels...")
mat_path = f"/kaggle/input/datasets/abdelrahmanyousryyu/bci-competition-iv-data-sets-2a-true-labels/A0{subject}E.mat"


file_to_load = mat_path
mat_data = sio.loadmat(file_to_load)

# proccessing file data
if 'classlabel' in mat_data:
    true_trial_labels = mat_data['classlabel'].flatten()
elif 'labels' in mat_data:
    true_trial_labels = mat_data['labels'].flatten()
else:
    true_trial_labels = None
    for key, val in mat_data.items():
        if isinstance(val, np.ndarray) and val.flatten().shape[0] in [288, 43198]:
            true_trial_labels = val.flatten()
            break
    if true_trial_labels is None:
        raise KeyError(f"Could not automatically parse labels. Keys available: {list(mat_data.keys())}")

if true_trial_labels.min() == 1:
    true_trial_labels = true_trial_labels - 1

print(f"Successfully loaded {len(true_trial_labels)} trial-level true keys.")

mean_eval = np.mean(X_train, axis=(0, 2), keepdims=True)
std_eval = np.std(X_train, axis=(0, 2), keepdims=True) + 1e-8


# STEP 3: EXTRACT EVALUATION WINDOWS FROM GDF

print(f"\n=== Processing Subject {subject} Evaluation GDF ===")
eval_file = f'/kaggle/input/datasets/abdelrahmanyousryyu/bci-comp2a/A0{subject}E.gdf'

eval_eeg, eval_events, eval_dict = preprocessor.process_file(eval_file, is_training=False)

# And use it here
X_eval, _, t_eval = BCICausalPreprocessor.generate_causal_windows(eval_eeg, eval_events, eval_dict, is_training=False)

# Normalize Evaluation windows using TRAINING stats
print("Applying global training normalization to evaluation data...")
X_eval_norm = (X_eval - mean_eval) / std_eval


# STEP 4: UNSCRAMBLE LABELS USING THE CLOCK RESET

print("Reconstructing trial indices from the relative timeline clock...")

trial_mappings = np.zeros(len(t_eval), dtype=np.int64)
current_trial_idx = 0

for i in range(1, len(t_eval)):
    if t_eval[i] < t_eval[i-1] or t_eval[i] == 0.0:
        current_trial_idx += 1
    
    if current_trial_idx >= len(true_trial_labels):
        current_trial_idx = len(true_trial_labels) - 1
        
    trial_mappings[i] = current_trial_idx

y_eval_true = true_trial_labels[trial_mappings]

print(f"Labels successfully aligned! Verification Shape: {y_eval_true.shape}")
print(f"Total unique trials reconstructed from timeline: {len(np.unique(trial_mappings))}/288")

eval_loader = DataLoader(
    TensorDataset(torch.tensor(X_eval_norm, dtype=torch.float32), torch.tensor(y_eval_true, dtype=torch.long)),
    batch_size=64, shuffle=False
)

BCICausalPreprocessor.visualize(*BCICausalPreprocessor.evaluate(model, device, eval_loader, true_trial_labels, trial_mappings))

# -----------------------------------------------------------------------------------------------------

# if os.path.isdir(mat_path):
#     mat_files = [f for f in os.listdir(mat_path) if f.endswith('.mat')]
#     if not mat_files:
#         raise FileNotFoundError(f"No .mat file found inside directory {mat_path}")
#     file_to_load = os.path.join(mat_path, mat_files[0])
# else:

