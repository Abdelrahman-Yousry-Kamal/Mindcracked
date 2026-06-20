import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from preprocessing import BCICausalPreprocessor
from PINN import NM_PINN


# ==========================================
# 0. USER INPUT & DEVICE SETUP
# ==========================================
while True:
    try:
        subject_id = int(input("Enter the Subject Number to train (1-9): "))
        if 1 <= subject_id <= 9:
            print(f"\nSubject {subject_id} selected. Initializing PINN pipeline...")
            break
        else:
            print("Error: Invalid input. Please enter a number between 1 and 9.")
    except ValueError:
        print("Error: Invalid input. Please enter a valid integer.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training PINN on 100% of Session T data on device: {device}\n")


# ==========================================
# 1. LOAD DATA & PREPROCESSING
# ==========================================
print(f"Loading and preprocessing Subject {subject_id} training data...")

train_gdf_file = f'/kaggle/input/datasets/abdelrahmanyousryyu/bci-comp2a/A0{subject_id}T.gdf'

preprocessor = BCICausalPreprocessor(lowcut=8.0, highcut=30.0)
eeg_train, events_train, dict_train = preprocessor.process_file(train_gdf_file, is_training=True)
X_train, y_train, _ = BCICausalPreprocessor.generate_causal_windows(eeg_train, events_train, dict_train, is_training=True)

# Global per-channel normalization
train_mean = np.mean(X_train, axis=(0, 2), keepdims=True)
train_std = np.std(X_train, axis=(0, 2), keepdims=True) + 1e-8
X_train_norm = (X_train - train_mean) / train_std

# Persist normalization stats for evaluation
os.makedirs('/kaggle/working/', exist_ok=True)
np.savez(f'/kaggle/working/pinn_stats_sub{subject_id}.npz', 
         mean=train_mean, std=train_std, eog_weights=preprocessor.eog_weights)

print(f"Data prepared! Shape: {X_train_norm.shape}\n")


# ==========================================
# 2. DATALOADER
# ==========================================
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), 
                  torch.tensor(y_train, dtype=torch.long)), 
    batch_size=64, shuffle=True
)


# ==========================================
# 3. MODEL & OPTIMIZER (PINN-SPECIFIC)
# ==========================================
model = NM_PINN(num_channels=22, F1=8, D=2, num_classes=4, fs=250).to(device)

# Differential Learning Rates for Physics Parameters
physics_param_names = {'tau_E', 'tau_I', 'w_EE', 'w_EI', 'w_IE', 'w_II', 'P', 'Q'}
physics_params = [p for n, p in model.named_parameters() if any(x in n for x in physics_param_names)]
base_params = [p for n, p in model.named_parameters() if not any(x in n for x in physics_param_names)]

optimizer = optim.AdamW([
    {'params': base_params, 'lr': 1e-3},
    {'params': physics_params, 'lr': 1e-4}
], weight_decay=1e-3)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# ==========================================
# 4. TRAINING LOOP
# ==========================================
epochs = 350
max_lambda_wc = 0.002   # Wilson-Cowan ODE residual weight
lambda_ortho = 0.01     # Spatial orthogonality constraint weight

print(f"Starting PINN training for Subject {subject_id}...\n")

for epoch in range(epochs):
    model.train()
    running_ce_loss = running_wc_loss = running_ortho_loss = 0.0
    correct = total = 0

    # Dynamic lambda scaling with warmup
    current_lambda_wc = 0.0 if epoch < 10 else max_lambda_wc * ((epoch - 10) / (epochs - 10))

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass: model returns logits and regularization losses
        logits, wc_loss, ortho_loss = model(inputs)

        # Composite loss
        ce_loss = criterion(logits, labels)
        total_loss = ce_loss + (current_lambda_wc * wc_loss) + (lambda_ortho * ortho_loss)

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        running_ce_loss += ce_loss.item()
        running_wc_loss += wc_loss.item()
        running_ortho_loss += ortho_loss.item()

        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Logging
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | L_wc: {current_lambda_wc:.3f} | "
              f"CE: {running_ce_loss/len(train_loader):.4f} | "
              f"WC: {running_wc_loss/len(train_loader):.4f} | "
              f"Ortho: {running_ortho_loss/len(train_loader):.4f} | "
              f"Acc: {100 * correct / total:.2f}%")


# ==========================================
# 5. SAVE MODEL
# ==========================================
torch.save(model.state_dict(), f"/kaggle/working/pinn_subject{subject_id}.pth")
print(f"\nModel saved: pinn_subject{subject_id}.pth")