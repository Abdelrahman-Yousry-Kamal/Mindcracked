import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import load_data, load_source, verify_model
from utils.slim_student import SlimEEGStudent

def run(device, output_dir="experimental_results"):
    print("\n" + "="*57)
    print("ATTACK 3: KNOWLEDGE DISTILLATION — SLIM EEG STUDENT (BLACK-BOX)")
    print("Objective: Train a different EEG-optimised architecture to mimic")
    print("           source model decisions, triggering black-box verify route")
    print("="*57)

    train_loader, X_test, y_test = load_data(device)
    teacher = load_source(device)
    teacher.eval()
    
    student = SlimEEGStudent(num_channels=22, num_classes=4, fs=250).to(device)

    optimizer_kd = optim.Adam(student.parameters(), lr=2e-3, weight_decay=1e-4)
    ce_loss  = nn.CrossEntropyLoss()
    kl_loss  = nn.KLDivLoss(reduction='batchmean')
    T = 4.0

    def fgsm(model, X, y, eps=0.05):
        X_adv = X.clone().detach().requires_grad_(True)
        loss  = ce_loss(model(X_adv, fingerprint_mode=True), y)
        loss.backward()
        return (X + eps * X_adv.grad.sign()).detach()

    student.train()
    print("Distilling (25 epochs, temperature-softened + FGSM boundary divergence)...")
    for epoch in range(25):
        for X_batch, y_batch in train_loader:
            with torch.no_grad():
                t_log = teacher(X_batch, fingerprint_mode=True)
                t_probs = torch.softmax(t_log / T, dim=1)

            X_adv = fgsm(student, X_batch, y_batch, eps=0.05)
            optimizer_kd.zero_grad()

            s_log_probs = torch.log_softmax(student(X_batch, fingerprint_mode=True) / T, dim=1)
            loss_kd = kl_loss(s_log_probs, t_probs)
            loss_adv = ce_loss(student(X_adv, fingerprint_mode=True), y_batch)

            loss = 0.5 * loss_kd + 0.5 * loss_adv
            loss.backward()
            optimizer_kd.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d}/25 | KD={loss_kd.item():.4f} | ADV={loss_adv.item():.4f}")

    os.makedirs(output_dir, exist_ok=True)
    kd_path = os.path.join(output_dir, "attack_distilled_student.pth")
    torch.save(student.state_dict(), kd_path)

    sidecar = os.path.join(output_dir, "attack_distilled_student.json")
    with open(sidecar, "w") as f:
        json.dump({"arch": "SlimEEGStudent"}, f)

    print(f"\n✅ Saved to {kd_path}")
    
    student.eval()
    with torch.no_grad():
        logits = student(X_test, fingerprint_mode=True)
        acc = (logits.argmax(1) == y_test).float().mean().item() * 100
    
    print(f"📊 Accuracy of Slim EEG Student: {acc:.2f}%")

    print("\n🔍 Auditing Fingerprint Survivability...")
    verify_model(kd_path, device)
    return kd_path
