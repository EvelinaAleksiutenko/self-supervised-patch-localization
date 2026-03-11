import torch


def compute_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> dict[str, float]:
    """Localization metrics. pred/gt: (N, 2) float tensors (y, x)."""
    pred_f = pred.float()
    gt_f = gt.float()
    per_sample_ed = torch.sqrt(((pred_f - gt_f) ** 2).sum(dim=1))
    return {
        'mean_ed': per_sample_ed.mean().item(),
        'median_ed': per_sample_ed.median().item(),
        'mae_y': (pred_f[:, 0] - gt_f[:, 0]).abs().mean().item(),
        'mae_x': (pred_f[:, 1] - gt_f[:, 1]).abs().mean().item(),
        'acc_at_1': (per_sample_ed <= 1).float().mean().item(),
        'acc_at_2': (per_sample_ed <= 2).float().mean().item(),
        'acc_at_5': (per_sample_ed <= 5).float().mean().item(),
    }
