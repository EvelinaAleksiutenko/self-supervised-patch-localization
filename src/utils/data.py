import csv
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from src.helpers.candidate_script import ImagePatchDataset
from src.config.config import Config

log = logging.getLogger(__name__)


def source_transform(cfg: Config) -> T.Compose:
    """Standard preprocessing for source images."""
    return T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.Grayscale(),
        T.ToTensor(),
    ])


def patch_transform(cfg: Config) -> T.Compose:
    """Standard preprocessing for patch images."""
    return T.Compose([
        T.Resize((cfg.patch_size, cfg.patch_size)),
        T.Grayscale(),
        T.ToTensor(),
    ])


def _save_test_set(dataset: ImagePatchDataset, indices, out_dir: str) -> None:
    """Persist test source/patch images and GT coords to disk."""
    out = Path(out_dir)
    source_dir = out / 'source'
    patch_dir = out / 'patch'
    source_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out / 'coords.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'y_start', 'x_start'])

        for i, idx in enumerate(indices):
            sample = dataset[int(idx)]
            fname = f'{i:05d}.png'
            save_image(sample['source_image'], str(source_dir / fname))
            save_image(sample['patch'], str(patch_dir / fname))
            gt = sample['ground_truth_coords']
            writer.writerow([i, gt[0].item(), gt[1].item()])

    log.info('Test set saved to %s  (%d samples)', out_dir, len(indices))


def build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    """Build train, validation, and test splits; save test data to disk."""
    source_ds = torchvision.datasets.CIFAR100(
        root=cfg.data_root, train=True, download=True,
    )
    # Train set: random patches each epoch (preprocessing)
    train_ds = ImagePatchDataset(
        source_ds, patch_size=cfg.patch_size, img_size=cfg.img_size,
        deterministic=False,
    )
    # Val/test sets
    eval_ds = ImagePatchDataset(
        source_ds, patch_size=cfg.patch_size, img_size=cfg.img_size,
        deterministic=True, seed=cfg.seed,
    )

    candidate_indices = torch.load(cfg.indices_path, weights_only=False)
    n_total = len(candidate_indices)
    n_test = int(n_total * cfg.test_split)
    n_val = int(n_total * cfg.val_split)
    n_train = n_total - n_val - n_test

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_total)
    train_idx = candidate_indices[perm[:n_train].tolist()]
    val_idx = candidate_indices[perm[n_train:n_train + n_val].tolist()]
    test_idx = candidate_indices[perm[n_train + n_val:].tolist()]

    # Save test data to disk (deterministic patches) — skip if already exists
    if not (Path(cfg.test_data_dir) / 'coords.csv').exists():
        _save_test_set(eval_ds, test_idx, cfg.test_data_dir)

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
