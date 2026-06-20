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
