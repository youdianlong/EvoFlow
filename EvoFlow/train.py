import os
import argparse
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from data import ADLongitudinal3DDataset, DatasetSplitManager
from models import EvolutionGuidedEncoder, create_model
from engine import Trainer


def get_args():
    p = argparse.ArgumentParser(description='EvoFlow: Evolution-Guided Conditional Flow Matching')

    p.add_argument('--exp_name', type=str, default='evolution_guided_v2')

    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--clinical_csv', type=str, required=True)
    p.add_argument('--volume_size', type=int, nargs=3, default=[96, 112, 96])
    p.add_argument('--num_history', type=int, default=3)
    p.add_argument('--max_time_delta', type=float, default=120.0)
    p.add_argument('--random_target', action='store_true', default=True)
    p.add_argument('--split_ratio', type=float, nargs=3, default=[0.7, 0.15, 0.15])

    p.add_argument('--model_size', type=str, default='S', choices=['S', 'M', 'L'])
    p.add_argument('--cond_dim', type=int, default=256)
    p.add_argument('--use_checkpoint', action='store_true', default=True)

    p.add_argument('--ablate_difference', action='store_true')
    p.add_argument('--ablate_evolution', action='store_true')
    p.add_argument('--ablate_cross_attention', action='store_true')
    p.add_argument('--ablate_clinical', action='store_true')
    p.add_argument('--ablate_time_delta', action='store_true')

    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--grad_accum', type=int, default=4)

    p.add_argument('--sample_steps', type=int, default=50)
    p.add_argument('--cfg_scale', type=float, default=2.0)

    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--save_every', type=int, default=50)
    p.add_argument('--use_amp', action='store_true', default=True)
    p.add_argument('--output_dir', type=str, default='./output_evolution_v2')
    p.add_argument('--patience', type=int, default=50)

    return p.parse_args()


def build_dataloaders(args, output_dir):
    cache_dir = os.path.join(output_dir, 'mri_cache')
    split_file_path = os.path.join(output_dir, 'dataset_split.json')

    split_manager = DatasetSplitManager(
        args.data_path,
        tuple(args.split_ratio),
        random_seed=42,
        split_file=split_file_path,
    )

    train_dataset = ADLongitudinal3DDataset(
        args.data_path,
        args.clinical_csv,
        args.num_history,
        tuple(args.volume_size),
        mode='train',
        split_manager=split_manager,
        max_time_delta=args.max_time_delta,
        random_target=args.random_target,
        cache_dir=cache_dir,
        use_cache=True,
        preprocess_all=True,
    )

    val_dataset = ADLongitudinal3DDataset(
        args.data_path,
        args.clinical_csv,
        args.num_history,
        tuple(args.volume_size),
        mode='val',
        split_manager=split_manager,
        max_time_delta=args.max_time_delta,
        random_target=False,
        cache_dir=cache_dir,
        use_cache=True,
        preprocess_all=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_loader, val_loader


def build_models(args, device):
    model = create_model(
        args.model_size,
        args.num_history,
        args.cond_dim,
        args.use_checkpoint,
    ).to(device)

    encoder = EvolutionGuidedEncoder(
        num_history=args.num_history,
        out_dim=args.cond_dim,
        diff_feature_dim=128,
        evolution_dim=128,
        frame_feature_dim=64,
        use_difference=not args.ablate_difference,
        use_evolution=not args.ablate_evolution,
        use_cross_attention=not args.ablate_cross_attention,
        use_clinical=not args.ablate_clinical,
        use_time_delta=not args.ablate_time_delta,
    ).to(device)

    model_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"Model: {model_params/1e6:.2f}M, Encoder: {encoder_params/1e6:.2f}M")
    print(f"Total: {(model_params + encoder_params)/1e6:.2f}M")

    return model, encoder


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {args.exp_name}")
    print(f"Version: V2.1 - Fixed Standard Flow Matching")
    print(f"{'='*70}")
    print(f"\nAblation settings:")
    print(f"  - Difference encoding:  {'OFF' if args.ablate_difference else 'ON'}")
    print(f"  - Evolution encoding:   {'OFF' if args.ablate_evolution else 'ON'}")
    print(f"  - Cross-attention:      {'OFF' if args.ablate_cross_attention else 'ON'}")
    print(f"  - Clinical features:    {'OFF' if args.ablate_clinical else 'ON'}")
    print(f"  - Time delta:           {'OFF' if args.ablate_time_delta else 'ON'}")
    print(f"{'='*70}\n")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{args.exp_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    config = vars(args).copy()
    config['version'] = 'V2.1_fixed_standard_flow_matching'
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    train_loader, val_loader = build_dataloaders(args, output_dir)
    model, encoder = build_models(args, device)

    trainer = Trainer(model, encoder, train_loader, val_loader, args, device, output_dir, config)
    trainer.train()


if __name__ == '__main__':
    main()
