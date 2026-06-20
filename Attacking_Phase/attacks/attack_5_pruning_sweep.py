import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from .utils import load_data, load_source, evaluate_model, verify_model

def run(device, output_dir="experimental_results"):
    print("\n" + "="*57)
    print("ATTACK 5: L1 PRUNING SWEEP 10%→90% + ODE PERTURBATION (WHITE-BOX)")
    print("Objective: Find the minimum pruning ratio that breaks the fingerprint")
    print("           while keeping task accuracy acceptable (>50%)")
    print("="*57)

    train_loader, X_test, y_test = load_data(device)

    PRUNED_MODEL_DIR = os.path.join(output_dir, "pruned_models")
    os.makedirs(PRUNED_MODEL_DIR, exist_ok=True)

    pruning_results = []
    
    for ratio in [r/10.0 for r in range(1, 10)]:
        ratio_pct = int(ratio * 100)
        print(f"\n{'─'*55}")
        print(f"  Pruning ratio: {ratio_pct}%")
        print(f"{'─'*55}")

        model_p = load_source(device)

        params_to_prune = (
            (model_p.temp_conv_mu,   'weight'),
            (model_p.temp_conv_beta, 'weight'),
            (model_p.depthwise,      'weight'),
            (model_p.classifier,     'weight'),
        )
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=ratio,
        )
        for module, name in params_to_prune:
            prune.remove(module, name)

        ode_params = ['tau_E', 'tau_I', 'w_EE', 'w_EI', 'w_IE', 'w_II', 'P', 'Q']
        with torch.no_grad():
            for pname, param in model_p.named_parameters():
                if any(op == pname for op in ode_params):
                    noise_scale = ratio * 0.5 * param.abs().mean().item()
                    param.add_(torch.randn_like(param) * noise_scale)

        recovery_epochs = max(5, int(ratio * 15))
        optimizer_r = optim.Adam(model_p.parameters(), lr=5e-4)
        ce_r = nn.CrossEntropyLoss()
        model_p.train()
        for epoch in range(recovery_epochs):
            for X_batch, y_batch in train_loader:
                optimizer_r.zero_grad()
                ce_r(model_p(X_batch, fingerprint_mode=True), y_batch).backward()
                optimizer_r.step()

        save_path = os.path.join(PRUNED_MODEL_DIR, f"attack_pruned_{ratio_pct}.pth")
        torch.save(model_p.state_dict(), save_path)

        acc = evaluate_model(save_path, device)
        print(f"  📊 Accuracy: {acc:.2f}%")
        print(f"  💾 Saved → {save_path}")

        score, mut_dev = verify_model(save_path, device)
        
        detected = (score >= 0.50 or mut_dev < 0.45)
        verdict = "❌ FP Survived" if detected else "✅ FP Erased"
        print(f"  🔒 FP score: {score*100:.1f}% | Mutation Dev: {mut_dev*100:.1f}% | {verdict}")

        pruning_results.append((ratio_pct, acc, score*100, mut_dev*100, verdict))

    best = None
    for entry in pruning_results:
        ratio_pct, acc, score, mut_dev, verdict = entry
        if acc > 50 and 'Erased' in verdict:
            if best is None or score < best[2]:
                best = entry

    if best:
        best_ratio_pct, best_acc, best_score, best_mut_dev, _ = best
        best_ratio = best_ratio_pct / 100.0
        print(f"\n🏆 Best ratio: {best_ratio_pct}% | Acc: {best_acc:.2f}% | FP Score: {best_score:.1f}%")
        print(f"   Rebuilding best model with full pipeline...")

        model_best = load_source(device)
        params_to_prune = (
            (model_best.temp_conv_mu,   'weight'),
            (model_best.temp_conv_beta, 'weight'),
            (model_best.depthwise,      'weight'),
            (model_best.classifier,     'weight'),
        )
        prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=best_ratio)
        for module, name in params_to_prune:
            prune.remove(module, name)

        with torch.no_grad():
            for pname, param in model_best.named_parameters():
                if any(op == pname for op in ode_params):
                    noise_scale = best_ratio * 0.5 * param.abs().mean().item()
                    param.add_(torch.randn_like(param) * noise_scale)

        recovery_epochs = max(5, int(best_ratio * 15))
        optimizer_best = optim.Adam(model_best.parameters(), lr=5e-4)
        model_best.train()
        for epoch in range(recovery_epochs):
            for X_batch, y_batch in train_loader:
                optimizer_best.zero_grad()
                ce_r(model_best(X_batch, fingerprint_mode=True), y_batch).backward()
                optimizer_best.step()

        best_path = os.path.join(PRUNED_MODEL_DIR, "attack_pruned_best.pth")
        torch.save(model_best.state_dict(), best_path)
        print(f"   💾 Best model → {best_path}")

        best_score_raw, best_mut_dev_raw = verify_model(best_path, device)
        best_detected = (best_score_raw >= 0.50 or best_mut_dev_raw < 0.45)
        best_verdict = "❌ FP Survived" if best_detected else "✅ FP Erased"
        print(f"  🔒 FP score: {best_score_raw*100:.1f}% | Mutation Dev: {best_mut_dev_raw*100:.1f}% | {best_verdict}")
        return best_path
    else:
        print("\n⚠️  No ratio achieved both FP erasure and accuracy > 50%")
        return None
