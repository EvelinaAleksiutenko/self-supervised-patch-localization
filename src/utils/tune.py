"""Hyperparameter tuning with Optuna for patch localizer."""

from __future__ import annotations

import argparse
import json
import logging
import random

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalFileOpenLock, JournalFileStorage, JournalStorage
import torch
from torch.utils.data import DataLoader

from src.config.config import Config
from src.utils.data import build_dataloaders
from src.utils.losses import ncc_loss
from src.utils.metrics import compute_metrics
from src.utils.model import SiamesePatchLocalizer, extract_region
from src.utils.train import seed_everything

log = logging.getLogger(__name__)



def create_config_from_trial(trial: optuna.Trial, base_cfg: Config) -> Config:
    """Sample hyperparameters for a single Optuna trial."""
    cfg = Config(
        img_size=base_cfg.img_size,
        patch_size=base_cfg.patch_size,
        indices_path=base_cfg.indices_path,
        data_root=base_cfg.data_root,
        val_split=base_cfg.val_split,
        test_split=base_cfg.test_split,
        test_data_dir=base_cfg.test_data_dir,
        num_workers=base_cfg.num_workers,

        enc_out_channels=trial.suggest_categorical('enc_out_channels', [16, 32, 48, 64]),
        temperature=trial.suggest_float('temperature', 1.0, 30.0, log=True),

        batch_size=trial.suggest_categorical('batch_size', [64, 128]),
        lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        warmup_epochs=trial.suggest_int('warmup_epochs', 1, 5),
        grad_clip=trial.suggest_float('grad_clip', 0.5, 5.0),

        epochs=base_cfg.tune_epochs,
        early_stopping=base_cfg.early_stopping,
        save_every=0,

        checkpoint_path=base_cfg.checkpoint_path,
        seed=base_cfg.seed,
        device=base_cfg.device,

        wandb_enabled=False,
    )
    return cfg


def objective(
    trial: optuna.Trial,
    base_cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> float:
    """Train one trial and return best validation mean Euclidean distance."""
    cfg = create_config_from_trial(trial, base_cfg)
    device = torch.device(cfg.device)

    seed_everything(cfg.seed)
    model = SiamesePatchLocalizer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs - cfg.warmup_epochs, 1), eta_min=1e-6,
    )

    best_mean_ed = float('inf')
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        # Linear warmup
        if epoch <= cfg.warmup_epochs:
            warmup_lr = cfg.lr * epoch / cfg.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # Train
        model.train()
        for batch in train_loader:
            source = batch['source_image'].to(device)
            patch = batch['patch'].to(device)
            coords, _ = model(source, patch)
            region = extract_region(source, coords, cfg)
            loss = ncc_loss(region, patch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

        if epoch > cfg.warmup_epochs:
            scheduler.step()

        # Validate
        model.eval()
        all_preds, all_gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                source = batch['source_image'].to(device)
                patch = batch['patch'].to(device)
                gt = batch['ground_truth_coords'].to(device)
                coords, _ = model(source, patch)
                all_preds.append(coords.cpu())
                all_gts.append(gt.cpu())

        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_gts))
        mean_ed = metrics['mean_ed']

        # Report to Optuna for pruning
        trial.report(mean_ed, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping
        if mean_ed < best_mean_ed:
            best_mean_ed = mean_ed
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping:
                break

    return best_mean_ed



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument('--n-trials', type=int, default=30,
                        help='Number of Optuna trials (default: 30)')
    parser.add_argument('--tune-epochs', type=int, default=25,
                        help='Max epochs per trial (default: 25)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--study-name', type=str, default='patch-localizer-tuning')
    parser.add_argument('--storage', type=str, default='tuning_journal.log',
                        help='Path to Optuna journal file for persistent storage (default: tuning_journal.log)')
    parser.add_argument('--output', type=str, default='best_params.json',
                        help='Path to save best hyperparameters (default: best_params.json)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_cfg = Config(
        seed=args.seed,
        wandb_enabled=False,
        tune_epochs=args.tune_epochs,
    )
    if args.device:
        base_cfg.device = args.device

    seed_everything(base_cfg.seed)

    # Build data loaders once — shared across all trials
    train_loader, val_loader = build_dataloaders(base_cfg)
    log.info("Train: %d | Val: %d", len(train_loader.dataset), len(val_loader.dataset))

    sampler = TPESampler(seed=base_cfg.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    storage = JournalStorage(JournalFileStorage(
        args.storage, lock_obj=JournalFileOpenLock(args.storage),
    ))

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, base_cfg, train_loader, val_loader),
        n_trials=args.n_trials,
    )

    best = study.best_trial
    log.info("Best trial #%d — mean_ed: %.4f px", best.number, best.value)
    log.info("Best params: %s", best.params)

    result = {
        'best_trial': best.number,
        'best_mean_ed': best.value,
        'best_params': best.params,
    }
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    log.info("Best params saved to %s", args.output)


    print("\n=== Top 5 trials ===")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    for t in trials_sorted[:5]:
        print(f"  Trial {t.number:3d} | mean_ed={t.value:.4f} | {t.params}")


    p = best.params
    print("\n=== Suggested Config overrides ===")
    print(f"  enc_out_channels = {p['enc_out_channels']}")
    print(f"  temperature      = {p['temperature']:.4f}")
    print(f"  batch_size       = {p['batch_size']}")
    print(f"  lr               = {p['lr']:.6f}")
    print(f"  weight_decay     = {p['weight_decay']:.6f}")
    print(f"  warmup_epochs    = {p['warmup_epochs']}")
    print(f"  grad_clip        = {p['grad_clip']:.2f}")

    print(f"\nFull study persisted in: {args.storage}")
    print("Re-run the same command to resume, or load in Python:")
    print(f"  study = optuna.load_study('{args.study_name}',")
    print(f"      storage=JournalStorage(JournalFileStorage('{args.storage}'))")
    print(f"  print(study.trials_dataframe())")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    main()
