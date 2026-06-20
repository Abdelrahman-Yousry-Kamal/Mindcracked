import os
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import load_data, load_source, evaluate_model, verify_model

def run(device, output_dir="experimental_results"):
    print("\n" + "="*57)
    print("ATTACK 1: PHYSICS-ANCHORED FINE-TUNING (WHITE-BOX)")
    print("Objective: Shift ODE physics parameters (tau, w_EE/EI/IE/II)")
    print("           to relocate the decision boundary geometry,")
    print("           invalidating stored adversarial trajectory predictions")
    print("="*57)

    train_loader, X_test, y_test = load_data(device)
    model_ft = load_source(device)

    physics_params = ['tau_E', 'tau_I', 'w_EE', 'w_EI', 'w_IE', 'w_II', 'P', 'Q']
    phys_p, other_p = [], []
    for name, p in model_ft.named_parameters():
        (phys_p if any(pp in name for pp in physics_params) else other_p).append(p)

    optimizer = optim.Adam([
        {'params': phys_p,  'lr': 5e-3},
        {'params': other_p, 'lr': 1e-3},
    ])
    ce_loss = nn.CrossEntropyLoss()

    source_ref = load_source(device); source_ref.eval()

    model_ft.train()
    print("Training (25 epochs, physics-targeted LR)...")
    for epoch in range(25):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model_ft(X_batch, fingerprint_mode=True)
            loss_ce = ce_loss(logits, y_batch)

            with torch.no_grad():
                ref_logits = source_ref(X_batch, fingerprint_mode=True)
                ref_probs  = torch.softmax(ref_logits, dim=1)

            atk_log_probs = torch.log_softmax(logits, dim=1)
            loss_shift = -torch.mean(torch.sum(ref_probs * atk_log_probs, dim=1))

            loss = loss_ce + 0.5 * loss_shift
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d}/25 | CE={loss_ce.item():.4f} | Shift={loss_shift.item():.4f}")

    os.makedirs(output_dir, exist_ok=True)
    ft_path = os.path.join(output_dir, "attack_finetuned.pth")
    torch.save(model_ft.state_dict(), ft_path)
    print(f"\n✅ Saved to {ft_path}")
    
    acc = evaluate_model(ft_path, device)
    print(f"📊 Accuracy after Physics-Anchored Fine-Tuning: {acc:.2f}%")
    
    print("\n🔍 Auditing Fingerprint Survivability...")
    verify_model(ft_path, device)
    return ft_path
