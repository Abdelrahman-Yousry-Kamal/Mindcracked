import argparse
import torch

from attacks import (
    attack_1_physics_finetuning,
    attack_2_trajectory_poisoning,
    attack_3_knowledge_distillation,
    attack_4_pinn_distribution,
    attack_5_pruning_sweep
)

def main():
    parser = argparse.ArgumentParser(description="ADV-TRA-EEG-BCI Attack Execution Engine")
    parser.add_argument("--attack", type=str, choices=["1", "2", "3", "4", "5", "all"], 
                        default="all", help="Select which attack to run")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--output_dir", type=str, default="experimental_results", help="Directory to save attack artifacts")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"✅ Initialized Attack Engine. Target Device: {device}")
    
    if args.attack in ["1", "all"]:
        attack_1_physics_finetuning.run(device, args.output_dir)
        
    if args.attack in ["2", "all"]:
        attack_2_trajectory_poisoning.run(device, args.output_dir)
        
    if args.attack in ["3", "all"]:
        attack_3_knowledge_distillation.run(device, args.output_dir)
        
    if args.attack in ["4", "all"]:
        attack_4_pinn_distribution.run(device, args.output_dir)
        
    if args.attack in ["5", "all"]:
        attack_5_pruning_sweep.run(device, args.output_dir)

if __name__ == "__main__":
    main()
