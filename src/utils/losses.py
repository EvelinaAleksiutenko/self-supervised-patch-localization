import torch


def ncc_loss(
    region: torch.Tensor, patch: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the Normalized Cross-Correlation (NCC) loss between `region`
    and `patch`.

    For each image in the batch and for each channel, the tensors are
    mean-centered and the correlation between the corresponding H×W
    patches is computed. This produces one NCC value per (batch, channel)
    pair. The final loss is the mean of (1 - NCC) over all batch and
    channel entries.

    Args:
        region (torch.Tensor): Tensor of shape (B, C, H, W).
        patch (torch.Tensor): Tensor of shape (B, C, H, W).
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Scalar tensor containing the mean NCC loss.
    """
    r = region - region.mean(dim=[2, 3], keepdim=True)
    p = patch - patch.mean(dim=[2, 3], keepdim=True)
    ncc = (r * p).sum(dim=[2, 3]) / (
        torch.sqrt((r ** 2).sum(dim=[2, 3]) * (p ** 2).sum(dim=[2, 3]) + eps)
    )
    return (1 - ncc).mean()
