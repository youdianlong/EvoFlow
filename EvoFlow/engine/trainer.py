import os
import math
import json

import torch
from tqdm import tqdm

from engine.losses import standard_flow_loss
from engine.validator import validate


class Trainer:

    def __init__(self, model, encoder, train_loader, val_loader, args, device, output_dir, config):
        self.model = model
        self.encoder = encoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        self.output_dir = output_dir
        self.config = config

        self.params = list(model.parameters()) + list(encoder.parameters())
        self.optimizer = torch.optim.AdamW(self.params, lr=args.lr, weight_decay=0.01)
        self.scheduler = self._build_scheduler()
        self.scaler = self._build_scaler()

        self.best_psnr = 0
        self.patience_counter = 0
        self.history_metrics = []
        self.global_step = 0

    def _build_scheduler(self):
        def lr_lambda(step):
            if step < self.args.warmup:
                return step / self.args.warmup
            total_steps = self.args.epochs * len(self.train_loader)
            return 0.5 * (1 + math.cos(
                math.pi * (step - self.args.warmup) / (total_steps - self.args.warmup)
            ))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _build_scaler(self):
        if not self.args.use_amp:
            return None
        return torch.amp.GradScaler(
            'cuda',
            init_scale=2**10,
            growth_factor=1.5,
            backoff_factor=0.5,
            growth_interval=2000,
        )

    def _prepare_batch(self, batch):
        history = batch['history_images'].to(self.device)
        target = batch['target_image'].to(self.device)

        time_delta = batch['time_delta'].to(self.device) if not self.args.ablate_time_delta else None

        structure_sequence = None
        evolution_rates = None
        cumulative_change = None
        if not self.args.ablate_evolution:
            if 'structure_sequence' in batch:
                structure_sequence = batch['structure_sequence'].to(self.device)
            if 'evolution_rates' in batch:
                evolution_rates = batch['evolution_rates'].to(self.device)
            if 'cumulative_change' in batch:
                cumulative_change = batch['cumulative_change'].to(self.device)

        cont = batch['continuous_features'].to(self.device) if not self.args.ablate_clinical else None
        cat = batch['categorical_features'].to(self.device) if not self.args.ablate_clinical else None

        return history, target, time_delta, structure_sequence, evolution_rates, cumulative_change, cont, cat

    def _train_one_epoch(self, epoch):
        self.model.train()
        self.encoder.train()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        epoch_loss = 0
        epoch_delta_mean = 0
        valid_batches = 0
        accumulated_steps = 0

        for batch_idx, batch in enumerate(pbar):
            history, target, time_delta, struct_seq, evo_rates, cum_change, cont, cat = \
                self._prepare_batch(batch)

            if torch.isnan(history).any() or torch.isnan(target).any():
                print(f"Warning: NaN in input at batch {batch_idx}, skipping")
                continue

            try:
                with torch.amp.autocast('cuda', enabled=self.args.use_amp):
                    cond = self.encoder(
                        history, time_delta,
                        struct_seq, evo_rates, cum_change,
                        cont, cat
                    )
                    loss, loss_dict = standard_flow_loss(self.model, target, history, cond)
                    loss = loss / self.args.grad_accum

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping")
                    continue

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulated_steps += 1

            except RuntimeError as e:
                print(f"RuntimeError at batch {batch_idx}: {e}")
                self.optimizer.zero_grad()
                if self.scaler is not None:
                    self.scaler.update()
                continue

            if accumulated_steps >= self.args.grad_accum:
                try:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                        if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                            self.scaler.step(self.optimizer)
                        else:
                            print(f"Warning: Invalid gradient at step {self.global_step}, skipping optimizer step")
                        self.scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                        if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                            self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1
                    accumulated_steps = 0

                except RuntimeError as e:
                    print(f"RuntimeError during optimizer step: {e}")
                    self.optimizer.zero_grad()
                    if self.scaler is not None:
                        self.scaler.update()
                    accumulated_steps = 0
                    continue

            epoch_loss += loss.item() * self.args.grad_accum
            epoch_delta_mean += loss_dict['delta_mean']
            valid_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item()*self.args.grad_accum:.4f}',
                'v_loss': f'{loss_dict["v_loss"]:.4f}',
                'Δ_gt': f'{loss_dict["delta_mean"]:.3f}',
                'Δ_pred': f'{loss_dict["pred_delta_mean"]:.3f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            avg_delta = epoch_delta_mean / valid_batches
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Avg |Δ| = {avg_delta:.4f}, "
                  f"Valid batches: {valid_batches}/{len(self.train_loader)}")
            return avg_loss
        else:
            print(f"Epoch {epoch}: No valid batches!")
            return None

    def _save_checkpoint(self, epoch, filename):
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'encoder': self.encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, os.path.join(self.output_dir, 'checkpoints', filename))

    def _save_best(self, epoch):
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'encoder': self.encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config,
        }, os.path.join(self.output_dir, 'checkpoints', 'best.pt'))

    def train(self):
        for epoch in range(self.args.epochs):
            avg_loss = self._train_one_epoch(epoch)
            if avg_loss is None:
                continue

            if (epoch + 1) % self.args.eval_every == 0:
                metrics = validate(
                    self.model, self.encoder, self.val_loader,
                    self.args, self.device, epoch, self.output_dir
                )

                print(f"  Val PSNR: {metrics['psnr']:.2f} ± {metrics.get('psnr_std', 0):.2f}, "
                      f"SSIM: {metrics['ssim']:.4f} ± {metrics.get('ssim_std', 0):.4f}")

                self.history_metrics.append({'epoch': epoch, 'loss': avg_loss, **metrics})

                with open(os.path.join(self.output_dir, 'metrics_history.json'), 'w') as f:
                    json.dump(self.history_metrics, f, indent=2)

                if metrics['psnr'] > self.best_psnr:
                    self.best_psnr = metrics['psnr']
                    self.patience_counter = 0
                    self._save_best(epoch)
                    print(f"New best! PSNR: {self.best_psnr:.2f}")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if (epoch + 1) % self.args.save_every == 0:
                self._save_checkpoint(epoch, f'epoch_{epoch:04d}.pt')

        self._save_final_results()

    def _save_final_results(self):
        final_results = {
            'exp_name': self.args.exp_name,
            'best_psnr': self.best_psnr,
            'version': 'V2.1_fixed_standard_flow_matching',
            'ablations': {
                'difference': self.args.ablate_difference,
                'evolution': self.args.ablate_evolution,
                'cross_attention': self.args.ablate_cross_attention,
                'clinical': self.args.ablate_clinical,
                'time_delta': self.args.ablate_time_delta,
            }
        }
        with open(os.path.join(self.output_dir, 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"EXPERIMENT COMPLETE: {self.args.exp_name}")
        print(f"Best PSNR: {self.best_psnr:.2f}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}")
