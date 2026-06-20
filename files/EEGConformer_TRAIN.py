import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import os
import sys
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from preprocessing import BCICausalPreprocessor, generate_causal_windows
from models.EEGConformer import EEGConformer


# ==========================================
# 0. SETUP & INITIALIZATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Initializing 9-Subject EEG-Conformer Pipeline on device: {device}\n")

# Tracking metrics across all subjects
all_subject_accuracies = []
global_true_labels = []
global_pred_labels = []

epochs = 200


# ==========================================
# 1. MULTI-SUBJECT TRAINING LOOP
# ==========================================
for subject_id in range(1, 10):
    print(f"\n{'='*60}")
    print(f"⚙️ PROCESSING SUBJECT {subject_id} / 9")
    print(f"{'='*60}")
    
    # File paths
    train_gdf = f'/kaggle/input/datasets/abdelrahmanyousryyu/bci-comp2a/A0{subject_id}T.gdf'
    eval_gdf = f'/kaggle/input/datasets/abdelrahmanyousryyu/bci-comp2a/A0{subject_id}E.gdf'
    mat_path = f'/kaggle/input/datasets/abdelrahmanyousryyu/bci-competition-iv-data-sets-2a-true-labels/A0{subject_id}E.mat'
    
    # ===== DATA PREPARATION (TRAIN) =====
    preprocessor = BCICausalPreprocessor(lowcut=8.0, highcut=30.0)
    
    print(f"Loading & Preprocessing Training Data (A0{subject_id}T.gdf)...")
    eeg_train, events_train, dict_train = preprocessor.process_file(train_gdf, is_training=True)
    X_train, y_train, _ = generate_causal_windows(eeg_train, events_train, dict_train, is_training=True)
    
    # Global normalization
    train_mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    train_std = np.std(X_train, axis=(0, 2), keepdims=True) + 1e-8
    X_train_norm = (X_train - train_mean) / train_std
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), 
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=64, shuffle=True
    )
    
    # ===== MODEL INITIALIZATION & TRAINING =====
    print(f"Initializing fresh EEG-Conformer model for Subject {subject_id}...")
    
    model = EEGConformer(
        n_chans=22, 
        n_outputs=4, 
        n_times=500, 
        drop_prob=0.5,
        num_layers=6,
        num_heads=10
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    print("Starting training loop...")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        scheduler.step()
            
        if (epoch + 1) % 100 == 0 or epoch == 0:
            train_acc = 100 * correct / total
            current_lr = scheduler.get_last_lr()[0]
            print(f"   Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.6f} | "
                  f"CE Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
    
    torch.save(model.state_dict(), f"/kaggle/working/eegconformer_baseline_subject{subject_id}.pth")
    
    # ===== DATA PREPARATION (EVALUATION) =====
    print(f"Loading True Labels (A0{subject_id}E.mat)...")
    if os.path.isdir(mat_path):
        mat_files = [f for f in os.listdir(mat_path) if f.endswith('.mat')]
        file_to_load = os.path.join(mat_path, mat_files[0])
    else:
        file_to_load = mat_path

    mat_data = sio.loadmat(file_to_load)
    
    if 'classlabel' in mat_data:
        true_trial_labels = mat_data['classlabel'].flatten()
    elif 'labels' in mat_data:
        true_trial_labels = mat_data['labels'].flatten()
    else:
        for key, val in mat_data.items():
            if isinstance(val, np.ndarray) and val.flatten().shape[0] in [288, 43198]:
                true_trial_labels = val.flatten()
                break
                
    if true_trial_labels.min() == 1:
        true_trial_labels = true_trial_labels - 1

    print(f"Loading & Preprocessing Evaluation Data (A0{subject_id}E.gdf)...")
    eeg_eval, events_eval, dict_eval = preprocessor.process_file(eval_gdf, is_training=False)
    X_eval, _, t_eval = generate_causal_windows(eeg_eval, events_eval, dict_eval, is_training=False)
    
    X_eval_norm = (X_eval - train_mean) / train_std
    
    # Reconstruct trial indices from timeline clock
    trial_mappings = np.zeros(len(t_eval), dtype=np.int64)
    current_trial_idx = 0
    for i in range(1, len(t_eval)):
        if t_eval[i] < t_eval[i-1] or t_eval[i] == 0.0:
            current_trial_idx += 1
        if current_trial_idx >= len(true_trial_labels):
            current_trial_idx = len(true_trial_labels) - 1
        trial_mappings[i] = current_trial_idx
        
    y_eval_true = true_trial_labels[trial_mappings]
    
    eval_loader = DataLoader(
        TensorDataset(torch.tensor(X_eval_norm, dtype=torch.float32), 
                      torch.tensor(y_eval_true, dtype=torch.long)),
        batch_size=64, shuffle=False
    )
    
    # ===== EVALUATION & SCORING =====
    model.eval()
    trial_logits_accumulator = {i: [] for i in range(len(true_trial_labels))}
    
    with torch.no_grad():
        window_idx = 0 
        for inputs, labels in eval_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            logits_cpu = logits.cpu().numpy()
            
            for batch_i in range(logits.size(0)):
                current_trial = trial_mappings[window_idx]
                trial_logits_accumulator[current_trial].append(logits_cpu[batch_i])
                window_idx += 1

    eval_correct = 0
    subject_true_labels = []
    subject_pred_labels = []

    for trial_id, logit_list in trial_logits_accumulator.items():
        if len(logit_list) == 0:
            continue
            
        mean_logits = np.mean(logit_list, axis=0)
        predicted_class = np.argmax(mean_logits)
        true_class = true_trial_labels[trial_id]
        
        subject_true_labels.append(true_class)
        subject_pred_labels.append(predicted_class)
        
        if predicted_class == true_class:
            eval_correct += 1

    final_holdout_acc = 100 * eval_correct / len(true_trial_labels)
    all_subject_accuracies.append(final_holdout_acc)
    
    global_true_labels.extend(subject_true_labels)
    global_pred_labels.extend(subject_pred_labels)
    
    print(f"✅ Subject {subject_id} Final Holdout Acc: {final_holdout_acc:.2f}%")
    
    del model, optimizer, X_train, X_train_norm, X_eval, X_eval_norm
    torch.cuda.empty_cache()


# ==========================================
# 2. FINAL RESULTS & VISUALIZATION
# ==========================================
overall_mean_acc = np.mean(all_subject_accuracies)

print(f"\n{'='*60}")
print(f"🏆 EEG-CONFORMER OVERALL BCI COMP IV-2a RESULTS")
print(f"{'='*60}")
for i, acc in enumerate(all_subject_accuracies):
    print(f"Subject {i+1}: {acc:.2f}%")
print(f"--------------------------------------------------")
print(f"GLOBAL MEAN ACCURACY: {overall_mean_acc:.2f}%")
print(f"{'='*60}\n")

# Confusion Matrix
class_names = ['Left Hand', 'Right Hand', 'Both Feet', 'Tongue']
cm = confusion_matrix(global_true_labels, global_pred_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names,
            annot_kws={"size": 14})

plt.title(f'Global EEG-Conformer Confusion Matrix (Mean Acc: {overall_mean_acc:.1f}%)', 
          fontsize=16, pad=15)
plt.xlabel('Predicted MI Class', fontsize=14, labelpad=10)
plt.ylabel('True MI Class', fontsize=14, labelpad=10)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
