import torch


def ncc_loss(
    region: torch.Tensor, patch: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    """
    NCC-based self-supervised loss: 1 - NCC(extracted_region, patch).

    NCC is invariant to global intensity offset and scale — more robust than
    L1/MSE for image matching, and the standard loss in registration literature
    (VoxelMorph, ANTs).

    Loss = 0  at the correct location (perfect structural match)
    Loss = 2  at worst case (perfectly anti-correlated)
    """
    r = region - region.mean(dim=[2, 3], keepdim=True)
    p = patch - patch.mean(dim=[2, 3], keepdim=True)
    ncc = (r * p).sum(dim=[2, 3]) / (
        torch.sqrt((r ** 2).sum(dim=[2, 3]) * (p ** 2).sum(dim=[2, 3]) + eps)
    )
    return (1 - ncc).mean()
