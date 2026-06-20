import argparse
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.models import get_model
from fingerprint_evaluator import evaluate_accuracy_direct_raw

def load_data(device):
    data_log = torch.load('data/bci_sub2a/allocated_data/data_log.pth', map_location=device, weights_only=False)
    train_ds = TensorDataset(data_log["X_train"].to(device), data_log["y_train"].to(device))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    X_test = data_log["X_attack"].to(device)
    y_test = data_log["y_attack"].to(device)
    return train_loader, X_test, y_test

def load_source(device):
    m = get_model('pinn', num_classes=4).to(device)
    m.load_state_dict(torch.load("model_path/bci_sub2a/source_model.pth", map_location=device, weights_only=False))
    return m

def evaluate_model(model_path, device):
    args = argparse.Namespace(
        dataset='bci_sub2a',
        data_path='./data',
        device=device,
        test_model_path=model_path,
        seed=42,
        num_classes=4,
        seq_length=250,
        num_channels=22
    )
    return evaluate_accuracy_direct_raw(args)

def verify_model(model_path, device):
    from fingerprint_evaluator import verify_pure_blackbox_trajectory
    from utils.adv_gen import verify_trajectory
    import tempfile
    import shutil

    # We need a staging dir to pass to the verifier logic since it expects a directory
    with tempfile.TemporaryDirectory() as staging_dir:
        staging_path = os.path.join(staging_dir, os.path.basename(model_path))
        shutil.copy2(model_path, staging_path)

        args = argparse.Namespace(
            dataset='bci_sub2a',
            data_path='./data',
            verify_target=staging_dir,
            device=device,
            trajectory_path='fingerprint_path',
            eps=0.1,
            num_trajectories=20,
            batch_size=32,
            seed=42,
            _current_verify_file=staging_path
        )

        try:
            # We attempt whitebox verify if it is a PINN model
            m = get_model('pinn', num_classes=4).to(device)
            m.load_state_dict(torch.load(staging_path, map_location=device, weights_only=False))
            m.eval()
            print("  [Verifier] Architecture matched (White-box PINN route).")
            return verify_trajectory(args, m)
        except Exception as e:
            # Fall back to black box verification
            print(f"  [Verifier] Fallback to black-box mode: {e}")
            raw_checkpoint = torch.load(staging_path, map_location=device, weights_only=False)
            return verify_pure_blackbox_trajectory(args, raw_checkpoint)
