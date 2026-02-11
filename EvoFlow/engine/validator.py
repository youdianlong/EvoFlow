import os

import torch
import numpy as np
from tqdm import tqdm

from engine.sampling import standard_flow_sample
from engine.metrics import calc_psnr_3d, calc_ssim_3d
from utils.visualization import save_comparison_3d


@torch.no_grad()
def validate(model, encoder, val_loader, args, device, epoch, output_dir):
    model.eval()
    encoder.eval()

    samples_dir = os.path.join(output_dir, f'samples_epoch{epoch:04d}')
    os.makedirs(samples_dir, exist_ok=True)

    all_psnr, all_ssim = [], []
    saved_samples = 0
    max_save = 5

    for batch in tqdm(val_loader, desc='Validating'):
        history = batch['history_images'].to(device)
        target = batch['target_image'].to(device)

        time_delta = batch['time_delta'].to(device) if not args.ablate_time_delta else None

        structure_sequence = None
        evolution_rates = None
        cumulative_change = None
        if not args.ablate_evolution:
            if 'structure_sequence' in batch:
                structure_sequence = batch['structure_sequence'].to(device)
            if 'evolution_rates' in batch:
                evolution_rates = batch['evolution_rates'].to(device)
            if 'cumulative_change' in batch:
                cumulative_change = batch['cumulative_change'].to(device)

        cont = batch['continuous_features'].to(device) if not args.ablate_clinical else None
        cat = batch['categorical_features'].to(device) if not args.ablate_clinical else None

        cond = encoder(
            history, time_delta,
            structure_sequence, evolution_rates, cumulative_change,
            cont, cat
        )

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            gen = standard_flow_sample(
                model, history, cond,
                num_steps=args.sample_steps // 2,
                cfg_scale=args.cfg_scale,
                device=device
            )

        gen_01 = (gen + 1) / 2
        target_01 = (target + 1) / 2

        for i in range(gen.shape[0]):
            psnr = calc_psnr_3d(gen_01[i:i+1], target_01[i:i+1])
            ssim = calc_ssim_3d(gen_01[i:i+1], target_01[i:i+1])
            all_psnr.append(psnr)
            all_ssim.append(ssim)

            if saved_samples < max_save:
                save_comparison_3d(
                    history[i], gen[i], target[i],
                    os.path.join(samples_dir, f'sample_{saved_samples}.png'),
                    {'psnr': psnr, 'ssim': ssim}
                )
                saved_samples += 1

    return {
        'psnr': np.mean(all_psnr),
        'ssim': np.mean(all_ssim),
        'psnr_std': np.std(all_psnr),
        'ssim_std': np.std(all_ssim),
        'n_samples': len(all_psnr),
    }
