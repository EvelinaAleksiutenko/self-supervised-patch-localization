from dataclasses import dataclass
import torch


@dataclass
class Config:

    img_size:        int   = 64
    patch_size:      int   = 16
    indices_path:    str   = 'train_val_indices.pt'
    data_root:       str   = './data'
    val_split:       float = 0.15
    test_split:      float = 0.15
    test_data_dir:   str   = 'test_data'
    num_workers:     int   = 2

    enc_out_channels: int   = 16
    temperature:     float = 1.8660  # soft-argmax sharpness

    batch_size:      int   = 128
    epochs:          int   = 50
    lr:              float = 7.31e-4
    weight_decay:    float = 1.5e-5
    early_stopping:  int   = 10      # patience: epochs to wait without improvement
    warmup_epochs:   int   = 4       # linear warmup before cosine schedule
    grad_clip:       float = 1.13    # max gradient norm
    save_every:      int   = 1       # save checkpoint every N epochs, (0 = best only)

    tune_epochs:     int   = 25      # max epochs per Optuna trial

    checkpoint_path: str   = 'model.pt' # TODO: change to 'checkpoints/model.pt' while training
    seed:            int   = 42
    device:          str   = 'cpu' # TODO: change to 'cuda' if torch.cuda.is_available() else 'cpu'

    wandb_project:   str   = 'patch-localization'
    wandb_run_name:  str   = ''
    wandb_entity:    str   = ''        # W&B team name
    wandb_enabled:   bool  = True
