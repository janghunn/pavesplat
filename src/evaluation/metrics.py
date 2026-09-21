from functools import cache

import torch
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    net = get_lpips(predicted.device)
    try:
        return net.forward(ground_truth, predicted, normalize=True)[:, 0, 0, 0]  # LPIPS_OOM_FALLBACK
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    _vals = []
    for _i in range(predicted.shape[0]):
        try:
            _v = net.forward(ground_truth[_i:_i+1], predicted[_i:_i+1], normalize=True)[:, 0, 0, 0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _v = get_lpips(torch.device("cpu")).forward(ground_truth[_i:_i+1].cpu(), predicted[_i:_i+1].cpu(), normalize=True)[:, 0, 0, 0]
        _vals.append(float(_v.reshape(-1)[0]))
        torch.cuda.empty_cache()
    return torch.tensor(_vals, device=predicted.device)


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)

@torch.no_grad()
def compute_depth_errors(
    ground_truths: Float[Tensor, "batch height width"],
    predicteds: Float[Tensor, "batch height width"],
    eval_min_depth: float = 0.5,
    eval_max_depth: float = 10.0,
) -> Float[Tensor, "7 batch"]:
    """
    Computes the Relative Depth Error (REL) between ground truth and predicted depth maps.
    """
    metrics_all = []
    for i,(ground_truth, predicted) in enumerate(zip(ground_truths, predicteds)):
        mask =  (ground_truth > 0) & (ground_truth > eval_min_depth) & (ground_truth < eval_max_depth)
        metrics = list(compute_errors(ground_truth[mask], predicted[mask]))
        metrics = torch.stack(metrics)
        metrics_all.append(metrics)
        
    metrics = torch.stack(metrics_all, dim=-1)
    return metrics

@torch.no_grad()
def compute_depth_errors_aligned(
    ground_truths: Float[Tensor, "batch height width"],
    predicteds: Float[Tensor, "batch height width"],
    eval_min_depth: float = 0.5,
    eval_max_depth: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """Median-scale-aligned depth errors (scale-invariant diagnostic).

    Per-image, predicted depth is rescaled by ``median(gt)/median(pred)`` over the
    valid mask before the error is computed. This removes a global depth-scale
    offset between predicted (rendered) and GT depth, isolating *shape* error from
    a *scale* shift. Returns ``(abs_rel_aligned, a1_aligned)`` stacked over batch.
    Used only as a diagnostic column; the raw (unaligned) protocol is unchanged.
    """
    absrel_all = []
    a1_all = []
    for ground_truth, predicted in zip(ground_truths, predicteds):
        mask = (ground_truth > 0) & (ground_truth > eval_min_depth) & (ground_truth < eval_max_depth)
        gt = ground_truth[mask]
        pred = predicted[mask]
        scale = torch.median(gt) / torch.median(pred).clamp_min(1e-8)
        abs_rel, _, _, _, a1, _, _ = compute_errors(gt, pred * scale)
        absrel_all.append(abs_rel)
        a1_all.append(a1)
    return torch.stack(absrel_all), torch.stack(a1_all)

@torch.no_grad()
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3