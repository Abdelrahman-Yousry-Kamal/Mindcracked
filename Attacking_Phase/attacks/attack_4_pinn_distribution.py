import os
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import load_data, load_source, evaluate_model, verify_model

def run(device, output_dir="experimental_results"):
    print("\n" + "="*57)
    print("ATTACK 4: TARGETED PINN DISTRIBUTION ATTACK (WHITE-BOX)")
    print("Objective: Use activation hooks on temp_conv_mu, temp_conv_beta,")
    print("           depthwise to maximise MMD divergence of PINN feature")
    print("           distributions — corrupting trajectory boundary alignment")
    print("="*57)

    train_loader, X_test, y_test = load_data(device)
    source_ref = load_source(device); source_ref.eval()
    model_pinn_atk = load_source(device)

    atk_feats = {}; ref_feats = {}

    def make_hook(store, key):
        def _h(m, inp, out): store[key] = out
        return _h

    TARGET_LAYERS = ('temp_conv_mu', 'temp_conv_beta', 'depthwise')
    atk_handles, ref_handles = [], []
    for name, module in model_pinn_atk.named_modules():
        if name in TARGET_LAYERS:
            atk_handles.append(module.register_forward_hook(make_hook(atk_feats, name)))
    for name, module in source_ref.named_modules():
        if name in TARGET_LAYERS:
            ref_handles.append(module.register_forward_hook(make_hook(ref_feats, name)))

    def mmd(x, y, sigma=1.0):
        x, y = x.flatten(1).float(), y.flatten(1).float()
        xx = torch.mm(x, x.t()); yy = torch.mm(y, y.t()); xy = torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        Kxx = torch.exp(-(rx.t() + rx - 2*xx) / (2*sigma**2))
        Kyy = torch.exp(-(ry.t() + ry - 2*yy) / (2*sigma**2))
        Kxy = torch.exp(-(rx.t() + ry   - 2*xy) / (2*sigma**2))
        return Kxx.mean() + Kyy.mean() - 2*Kxy.mean()

    optimizer_pa = optim.Adam(model_pinn_atk.parameters(), lr=5e-4)
    ce_loss = nn.CrossEntropyLoss()
    LAMBDA = 4.0

    model_pinn_atk.train()
    print("Training (20 epochs)...")
    for epoch in range(20):
        epoch_ce = epoch_mmd = 0.0
        for X_batch, y_batch in train_loader:
            with torch.no_grad():
                source_ref(X_batch, fingerprint_mode=True)
            snap_ref = {k: v.detach() for k, v in ref_feats.items()}

            logits = model_pinn_atk(X_batch, fingerprint_mode=True)
            loss_ce = ce_loss(logits, y_batch)

            loss_mmd = torch.tensor(0.0, device=device)
            for key in TARGET_LAYERS:
                if key in atk_feats and key in snap_ref:
                    loss_mmd = loss_mmd + mmd(atk_feats[key], snap_ref[key])

            loss = loss_ce - LAMBDA * loss_mmd
            optimizer_pa.zero_grad()
            loss.backward()
            optimizer_pa.step()

            epoch_ce  += loss_ce.item()
            epoch_mmd += loss_mmd.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d}/20 | CE={epoch_ce/len(train_loader):.4f} | MMD={epoch_mmd/len(train_loader):.4f}")

    for h in atk_handles + ref_handles:
        h.remove()

    os.makedirs(output_dir, exist_ok=True)
    pinn_path = os.path.join(output_dir, "attack_pinn_distribution.pth")
    torch.save(model_pinn_atk.state_dict(), pinn_path)
    print(f"\n✅ Saved to {pinn_path}")
    
    acc = evaluate_model(pinn_path, device)
    print(f"📊 Accuracy after Targeted PINN Distribution Attack: {acc:.2f}%")

    print("\n🔍 Auditing Fingerprint Survivability...")
    verify_model(pinn_path, device)
    return pinn_path
