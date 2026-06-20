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
