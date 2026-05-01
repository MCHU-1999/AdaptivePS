import torch
import torch.nn.functional as F

def metric_depth_loss(depth_pred, depth_gt, mask, max_depth=4.0, weight=None):
    depth_mask = torch.logical_and(depth_gt<=max_depth, depth_gt>0)
    depth_mask = torch.logical_and(depth_mask, mask)
    if depth_mask.sum() == 0:
        depth_loss = torch.tensor([0.]).mean().cuda()
    else:
        if weight is None:
            depth_loss = torch.mean(torch.abs((depth_pred - depth_gt)[depth_mask]))
        else:
            depth_loss = torch.mean((weight * torch.abs(depth_pred - depth_gt))[depth_mask])
    return depth_loss

def normal_loss(normal_pred, normal_gt, mask):
    normal_pred = F.normalize(normal_pred, dim=-1)
    normal_gt = F.normalize(normal_gt, dim=-1)
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1)[mask].mean()
    cos = (1. - torch.sum(normal_pred * normal_gt, dim=-1))[mask].mean()
    return l1, cos

def edge_aware_l1_loss(pred, target, mask, edge_guide, edge_boost=2.0, eps=1e-6):
    """
    Edge-aware L1 loss that upweights pixels near strong discontinuities.

    Args:
        pred: Prediction tensor with shape [H, W], [H, W, C], or flattened to [N] / [N, C].
        target: Ground-truth tensor with the same shape as pred.
        mask: Boolean validity mask with shape [H, W] or [N].
        edge_guide: Single-channel guidance map used to detect edges, typically depth or a normal magnitude map.
        edge_boost: Strength of edge weighting. Larger values put more emphasis on boundary pixels.
        eps: Numerical stability term.

    Returns:
        Scalar weighted L1 loss.
    """
    if edge_guide.dim() == 1:
        raise ValueError('edge_guide must be at least 2D so edge gradients can be computed')

    guide = edge_guide.float()
    if guide.dim() == 3 and guide.shape[0] == 1:
        guide = guide[0]
    elif guide.dim() == 3 and guide.shape[-1] == 1:
        guide = guide[..., 0]
    elif guide.dim() == 3 and guide.shape[-1] > 1:
        guide = guide.mean(dim=-1)

    # Finite-difference gradient magnitude on the guidance map.
    dx = torch.zeros_like(guide)
    dy = torch.zeros_like(guide)
    dx[:, 1:] = guide[:, 1:] - guide[:, :-1]
    dy[1:, :] = guide[1:, :] - guide[:-1, :]
    edge_strength = torch.sqrt(dx * dx + dy * dy + eps)
    edge_strength = edge_strength / edge_strength.max().clamp(min=eps)

    weight = 1.0 + edge_boost * edge_strength

    if pred.dim() > 1 and pred.shape[-1] > 1:
        per_pixel_err = torch.abs(pred - target).sum(dim=-1)
    else:
        per_pixel_err = torch.abs(pred - target)

    flat_mask = mask.reshape(-1)
    flat_weight = weight.reshape(-1)
    flat_err = per_pixel_err.reshape(-1)

    valid = flat_mask.bool()
    if valid.sum() == 0:
        return torch.tensor(0., device=pred.device, dtype=pred.dtype)

    return (flat_err[valid] * flat_weight[valid]).mean()