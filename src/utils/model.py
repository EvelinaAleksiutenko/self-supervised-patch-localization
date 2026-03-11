import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import Config


class Encoder(nn.Module):
    """
    Lightweight shared-weight CNN encoder.
    No striding or pooling — preserves 1:1 spatial correspondence between
    feature map coordinates and pixel coordinates.
    """

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SiamesePatchLocalizer(nn.Module):
    """
    Siamese correlation filter for self-supervised patch localization.

    Pipeline:
        source  (B, 1, 64, 64)
        patch   (B, 1, 16, 16)
            ↓ shared encoder (no striding)
        Fs      (B, C, 64, 64)
        Fp      (B, C, 16, 16)
            ↓ batch cross-correlation  [F.conv2d with groups=B]
        corr    (B, 1, 49, 49)         score(i,j) = similarity of patch at position (i,j)
            ↓ soft-argmax
        coords  (B, 2)                 (y_pred, x_pred) in pixel units [0, 48]
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.encoder = Encoder(cfg.enc_out_channels)
        self.temperature = cfg.temperature

    def forward(
        self, source: torch.Tensor, patch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Fs = self.encoder(source)
        Fp = self.encoder(patch)
        corr = self._batch_xcorr(Fs, Fp)
        coords = self._soft_argmax(corr)
        return coords, corr

    def _batch_xcorr(self, Fs: torch.Tensor, Fp: torch.Tensor) -> torch.Tensor:
        """
        Batch-wise cross-correlation via grouped convolution.
        Each sample's patch features slide over its own source features.
        """
        B, C, Hs, Ws = Fs.shape
        _, _, Hp, Wp = Fp.shape
        out = F.conv2d(
            Fs.view(1, B * C, Hs, Ws),
            Fp.view(B, C, Hp, Wp),
            groups=B,
        )
        return out.view(B, 1, out.shape[2], out.shape[3])

    def _soft_argmax(self, corr: torch.Tensor) -> torch.Tensor:
        """
        Differentiable argmax:
            p(i,j) = softmax(temperature * corr)
            y_pred = Σ_ij  i · p(i,j)
            x_pred = Σ_ij  j · p(i,j)
        """
        B, _, H, W = corr.shape
        logits = corr.view(B, -1) * self.temperature
        prob = F.softmax(logits, dim=-1).view(B, H, W)
        ys = torch.arange(H, dtype=torch.float32, device=corr.device)
        xs = torch.arange(W, dtype=torch.float32, device=corr.device)
        y_pred = (prob * ys.view(1, H, 1)).sum(dim=[1, 2])
        x_pred = (prob * xs.view(1, 1, W)).sum(dim=[1, 2])
        return torch.stack([y_pred, x_pred], dim=1)


def extract_region(
    source: torch.Tensor,
    coords: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """
    Differentiably extract a patch_size×patch_size region from source at predicted coords.

    Uses F.grid_sample (spatial transformer) so gradients flow back through coords
    into the network — this is what makes the NCC loss train the localizer.

    source : (B, 1, img_size, img_size)
    coords : (B, 2)  — (y, x) top-left corner in pixel units, float
    returns: (B, 1, patch_size, patch_size)
    """
    device = source.device
    P = cfg.patch_size

    gi, gj = torch.meshgrid(
        torch.arange(P, dtype=torch.float32, device=device),
        torch.arange(P, dtype=torch.float32, device=device),
        indexing='ij',
    )

    y_abs = coords[:, 0:1, None] + gi.unsqueeze(0)
    x_abs = coords[:, 1:2, None] + gj.unsqueeze(0)

    # Normalize to [-1, 1] — required by grid_sample with align_corners=True
    y_norm = y_abs / (cfg.img_size - 1) * 2 - 1
    x_norm = x_abs / (cfg.img_size - 1) * 2 - 1

    grid = torch.stack([x_norm, y_norm], dim=-1)  # (B, P, P, 2) — x first
    return F.grid_sample(
        source, grid, mode='bilinear', padding_mode='border', align_corners=True,
    )
