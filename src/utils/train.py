"""Training orchestrator for self-supervised patch localization."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config.config import Config
from src.utils.data import build_dataloaders
from src.utils.losses import ncc_loss
from src.utils.metrics import compute_metrics
from src.utils.model import SiamesePatchLocalizer, extract_region
import wandb


log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Fix all random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class Trainer:
    """Encapsulates model, optimizer, scheduler, and training/validation logic."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.model = SiamesePatchLocalizer(cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-6,
        )

        self.best_val_loss = float('inf')
        self.best_mean_ed = float('inf')
        self.best_epoch = 0
        self._patience_counter = 0
        self._use_wandb = cfg.wandb_enabled and wandb is not None

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


    def train_one_epoch(self, loader: DataLoader) -> float:
        """Self-supervised training: predict coords → extract region → NCC loss (no GT used)."""
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            source = batch['source_image'].to(self.device)
            patch = batch['patch'].to(self.device)

            coords, _ = self.model(source, patch)
            region = extract_region(source, coords, self.cfg)
            loss = ncc_loss(region, patch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * source.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Compute NCC loss + localization metrics against GT coords (monitoring only)."""
        self.model.eval()
        total_ncc = 0.0
        all_preds: list[torch.Tensor] = []
        all_gts: list[torch.Tensor] = []

        for batch in loader:
            source = batch['source_image'].to(self.device)
            patch = batch['patch'].to(self.device)
            gt = batch['ground_truth_coords'].to(self.device)

            coords, _ = self.model(source, patch)
            region = extract_region(source, coords, self.cfg)

            total_ncc += ncc_loss(region, patch).item() * source.size(0)
            all_preds.append(coords.cpu())
            all_gts.append(gt.cpu())

        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_gts))
        metrics['loss'] = total_ncc / len(loader.dataset)
        return metrics


    def _save_checkpoint(self, epoch: int, val_loss: float, suffix: str = '') -> None:
        path = self.cfg.checkpoint_path
        if suffix:
            base, ext = path.rsplit('.', 1) if '.' in path else (path, 'pt')
            path = f'{base}_{suffix}.{ext}'
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'val_loss': val_loss,
                'cfg': self.cfg,
            },
            path,
        )


    def _init_wandb(self) -> None:
        if self.cfg.wandb_enabled and wandb is None:
            log.warning("wandb not installed — logging disabled")
        if self._use_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity or None,
                name=self.cfg.wandb_run_name or None,
                config={
                    'enc_out_channels': self.cfg.enc_out_channels,
                    'temperature': self.cfg.temperature,
                    'batch_size': self.cfg.batch_size,
                    'lr': self.cfg.lr,
                    'weight_decay': self.cfg.weight_decay,
                    'epochs': self.cfg.epochs,
                    'early_stopping': self.cfg.early_stopping,
                    'n_params': self.n_params,
                },
            )

    def _log_wandb(self, epoch: int, train_loss: float, val_metrics: dict[str, float]) -> None:
        if not self._use_wandb:
            return
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'val/loss': val_metrics['loss'],
            'val/mean_ed': val_metrics['mean_ed'],
            'val/median_ed': val_metrics['median_ed'],
            'val/mae_y': val_metrics['mae_y'],
            'val/mae_x': val_metrics['mae_x'],
            'val/acc_at_1': val_metrics['acc_at_1'],
            'val/acc_at_2': val_metrics['acc_at_2'],
            'val/acc_at_5': val_metrics['acc_at_5'],
            'lr': self.current_lr,
        }, step=epoch)


    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self._init_wandb()
        log.info("Parameters : %s", f"{self.n_params:,}")
        log.info("Device     : %s", self.device)

        for epoch in range(1, self.cfg.epochs + 1):
            # Linear warmup
            if epoch <= self.cfg.warmup_epochs:
                warmup_lr = self.cfg.lr * epoch / self.cfg.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = warmup_lr

            train_loss = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['loss']

            # Step cosine scheduler only after warmup
            if epoch > self.cfg.warmup_epochs:
                self.scheduler.step()

            log.info(
                "Epoch %3d | train_loss %.5f | val_loss %.5f | val_MED %.3f px | lr %.2e",
                epoch, train_loss, val_loss, val_metrics['mean_ed'], self.current_lr,
            )
            self._log_wandb(epoch, train_loss, val_metrics)

            # Periodic checkpoint (every N epochs)
            if self.cfg.save_every > 0 and epoch % self.cfg.save_every == 0:
                self._save_checkpoint(epoch, val_loss, suffix=f'epoch{epoch:03d}')
                log.info("  ↳ periodic checkpoint saved  (epoch %d)", epoch)

            # Early stopping: improve if val_loss OR mean_ed improved
            improved = False
            if val_loss < self.best_val_loss or val_metrics['mean_ed'] < self.best_mean_ed:
                improved = True
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                if val_metrics['mean_ed'] < self.best_mean_ed:
                    self.best_mean_ed = val_metrics['mean_ed']
                self.best_epoch = epoch
                self._patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
                log.info(
                    "  ↳ checkpoint saved  (val_loss=%.5f, MED=%.3f px)",
                    self.best_val_loss, self.best_mean_ed,
                )
                if self._use_wandb:
                    wandb.run.summary['best_val_loss'] = self.best_val_loss
                    wandb.run.summary['best_mean_ed'] = self.best_mean_ed
                    wandb.run.summary['best_epoch'] = epoch

            if not improved:
                self._patience_counter += 1
                if self._patience_counter >= self.cfg.early_stopping:
                    log.info("Early stopping at epoch %d.", epoch)
                    break

        log.info("Best val_loss=%.5f  MED=%.3f px  (epoch %d)",
                 self.best_val_loss, self.best_mean_ed, self.best_epoch)
        log.info("Checkpoint : %s", self.cfg.checkpoint_path)

        if self._use_wandb:
            wandb.finish()



def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train patch localizer")
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--enc-out-channels', type=int)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--early-stopping', type=int)
    parser.add_argument('--save-every', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=str)
    parser.add_argument('--checkpoint-path', type=str)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str)
    parser.add_argument('--wandb-run-name', type=str)
    parser.add_argument('--wandb-entity', type=str)
    args = parser.parse_args()

    cfg = Config()
    arg_to_cfg = {
        'epochs': 'epochs',
        'batch_size': 'batch_size',
        'lr': 'lr',
        'enc_out_channels': 'enc_out_channels',
        'temperature': 'temperature',
        'early_stopping': 'early_stopping',
        'save_every': 'save_every',
        'seed': 'seed',
        'device': 'device',
        'checkpoint_path': 'checkpoint_path',
        'wandb_project': 'wandb_project',
        'wandb_run_name': 'wandb_run_name',
        'wandb_entity': 'wandb_entity',
    }
    for arg_name, cfg_name in arg_to_cfg.items():
        val = getattr(args, arg_name)
        if val is not None:
            setattr(cfg, cfg_name, val)
    if args.no_wandb:
        cfg.wandb_enabled = False
    return cfg


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    train_loader, val_loader = build_dataloaders(cfg)
    log.info(
        "Train: %d  |  Val: %d  |  Test saved to: %s",
        len(train_loader.dataset), len(val_loader.dataset), cfg.test_data_dir,
    )
    Trainer(cfg).fit(train_loader, val_loader)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    main()
