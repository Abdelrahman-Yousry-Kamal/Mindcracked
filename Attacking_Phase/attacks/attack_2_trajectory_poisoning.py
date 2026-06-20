import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .utils import load_data, load_source, evaluate_model, verify_model

def run(device, output_dir="experimental_results"):
    print("\n" + "="*57)
    print("ATTACK 2: ADVERSARIAL TRAJECTORY POISONING (WHITE-BOX)")
    print("Objective: Directly replay stored fingerprint trajectories")
    print("           and fine-tune the model to give DIFFERENT predictions")
    print("           on those exact paths — directly invalidates the verifier")
    print("="*57)

    train_loader, X_test, y_test = load_data(device)
    model_ft = load_source(device)
    optimizer = optim.Adam(model_ft.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()

    fp_base = "fingerprint_path/bci_sub2a/trajectory_20"
    tra_files = sorted(glob.glob(f"{fp_base}/*/tra_log.pth"))
    pred_files = sorted(glob.glob(f"{fp_base}/*/pred_log.pth"))

    if not tra_files:
        print("🚨 No fingerprint trajectories found.")
        return None

    print(f"📂 Found {len(tra_files)} fingerprint trajectory paths")

    all_tra, all_wrong_labels = [], []
    for tra_path, pred_path in zip(tra_files, pred_files):
        tra  = torch.cat(torch.load(tra_path,  map_location=device, weights_only=False))
        pred = torch.cat(torch.load(pred_path, map_location=device, weights_only=False))

        for i in range(len(tra)):
            correct_label = pred[i].item()
            wrong_label = (correct_label + 1) % 4
            all_tra.append(tra[i])
            all_wrong_labels.append(wrong_label)

    tra_tensor   = torch.stack(all_tra).to(device)
    wrong_tensor = torch.tensor(all_wrong_labels, dtype=torch.long).to(device)
    tra_ds       = TensorDataset(tra_tensor, wrong_tensor)
    tra_loader   = DataLoader(tra_ds, batch_size=32, shuffle=True)

    print(f"📊 Trajectory dataset: {len(tra_tensor)} samples")
    print("Training (20 epochs — trajectory misprediction + task preservation)...")

    model_ft.train()
    for epoch in range(20):
        clean_iter = iter(train_loader)
        for X_tra, y_wrong in tra_loader:
            optimizer.zero_grad()
            logits_tra = model_ft(X_tra, fingerprint_mode=True)
            loss_fp    = ce_loss(logits_tra, y_wrong)

            try:
                X_clean, y_clean = next(clean_iter)
            except StopIteration:
                clean_iter = iter(train_loader)
                X_clean, y_clean = next(clean_iter)
                
            logits_clean = model_ft(X_clean, fingerprint_mode=True)
            loss_clean   = ce_loss(logits_clean, y_clean)

            loss = 0.6 * loss_fp + 0.4 * loss_clean
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d}/20 | FP={loss_fp.item():.4f} | Clean={loss_clean.item():.4f}")

    os.makedirs(output_dir, exist_ok=True)
    ft_path = os.path.join(output_dir, "attack_finetuned2.pth")
    torch.save(model_ft.state_dict(), ft_path)
    print(f"\n✅ Saved to {ft_path}")
    
    acc = evaluate_model(ft_path, device)
    print(f"📊 Accuracy after Trajectory Poisoning: {acc:.2f}%")

    print("\n🔍 Auditing Fingerprint Survivability...")
    verify_model(ft_path, device)
    return ft_path
