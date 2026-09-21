"""Surface-normal evaluation against GT depth (ScanNet rendered-normal error).

Eval-only helpers used by ``model_wrapper.test_step`` behind ``test.eval_normal_error``.
ScanNet ships GT depth but no GT normals, so a per-pixel GT normal is derived from the
GT depth map (camera space). The model's rendered normal (``output.normal``) is a
WORLD-space, alpha-composited (hence non-unit) field; it is rotated into the target
camera frame and L2-normalised before comparison.

Two angular metrics are reported:
  * unoriented  acos(|<n_pred, n_gt>|)  — robust headline; invariant to the normal sign,
    which is the principled choice because the two normal sources need not share an
    orientation convention.
  * oriented    acos(<n_pred, n_gt>)    — secondary; both normals are first flipped into the
    camera-facing hemisphere (n_z <= 0 in OpenCV camera space) so the sign is comparable.

Unprojection uses ``coords [..,h,w,2] in [0,1]`` from ``sample_image_grid`` and
normalised 3x3 intrinsics broadcast over the pixel grid.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..geometry.projection import sample_image_grid, unproject


@torch.no_grad()
def gt_normal_from_depth(
    depth: Tensor,            # [V, H, W] metric depth (meters)
    intrinsics: Tensor,       # [V, 3, 3] normalised pinhole intrinsics
    edge_rel_thresh: float = 0.05,
    eps: float = 1e-8,
):
    """Derive a per-pixel camera-space GT normal from a depth map via central differences.

    Returns:
        normal_cam: [V, H, W, 3] camera-space unit normals (zeros where invalid).
        valid:      [V, H, W] bool mask (interior, finite, on-surface, no depth jump).
    """
    v, h, w = depth.shape
    device = depth.device
    coords, _ = sample_image_grid((h, w), device=device)        # [h, w, 2] in [0, 1]
    coords = coords.expand(v, h, w, 2)
    intr = intrinsics.view(v, 1, 1, 3, 3)
    points = unproject(coords, depth, intr)                      # [v, h, w, 3] camera space

    # Central-difference tangents (interior pixels only).
    t_x = points[:, :, 2:, :] - points[:, :, :-2, :]            # [v, h,   w-2, 3]
    t_y = points[:, 2:, :, :] - points[:, :-2, :, :]            # [v, h-2, w,   3]
    t_x_i = t_x[:, 1:-1, :, :]                                   # [v, h-2, w-2, 3]
    t_y_i = t_y[:, :, 1:-1, :]                                   # [v, h-2, w-2, 3]
    n_i = torch.cross(t_x_i, t_y_i, dim=-1)                      # [v, h-2, w-2, 3]
    n_i = F.normalize(n_i, dim=-1, eps=eps)

    normal_cam = depth.new_zeros((v, h, w, 3))
    normal_cam[:, 1:-1, 1:-1, :] = n_i

    # Depth-discontinuity mask: relative inverse-depth jump to any 4-neighbour.
    xi = 1.0 / depth.clamp(min=eps)                             # [v, h, w]
    rel = torch.zeros_like(xi)
    rel[:, 1:-1, 1:-1] = torch.stack([
        (xi[:, 1:-1, 1:-1] - xi[:, :-2, 1:-1]).abs(),           # up
        (xi[:, 1:-1, 1:-1] - xi[:, 2:, 1:-1]).abs(),           # down
        (xi[:, 1:-1, 1:-1] - xi[:, 1:-1, :-2]).abs(),           # left
        (xi[:, 1:-1, 1:-1] - xi[:, 1:-1, 2:]).abs(),           # right
    ], dim=0).max(dim=0).values / xi[:, 1:-1, 1:-1].abs().clamp(min=eps)
    no_jump = rel < edge_rel_thresh

    valid = torch.isfinite(normal_cam).all(dim=-1) & no_jump & (depth > eps)
    valid[:, :1, :] = False
    valid[:, -1:, :] = False
    valid[:, :, :1] = False
    valid[:, :, -1:] = False
    normal_cam = torch.where(valid.unsqueeze(-1), normal_cam, torch.zeros_like(normal_cam))
    return normal_cam, valid


@torch.no_grad()
def rendered_normal_to_camera(
    normal_world: Tensor,     # [V, 3, H, W] world-space (alpha-composited, non-unit)
    extrinsics_c2w: Tensor,   # [V, 4, 4] camera-to-world
    eps: float = 1e-8,
):
    """Rotate the world-space rendered normal into each view's camera frame and normalise.

    Returns:
        normal_cam: [V, H, W, 3] unit normals.
        coverage:   [V, H, W] pre-normalisation norm (≈0 where no surfel was hit).
    """
    v = normal_world.shape[0]
    r_w2c = torch.inverse(extrinsics_c2w[:, :3, :3])            # [V, 3, 3]
    n = normal_world.permute(0, 2, 3, 1)                        # [V, H, W, 3] world
    n_cam = torch.einsum("vij,vhwj->vhwi", r_w2c, n)           # [V, H, W, 3] camera
    coverage = torch.linalg.norm(n_cam, dim=-1)                # [V, H, W]
    n_cam = F.normalize(n_cam, dim=-1, eps=eps)
    return n_cam, coverage


def _face_camera(n: Tensor) -> Tensor:
    """Flip normals into the camera-facing hemisphere (OpenCV: n_z <= 0)."""
    flip = torch.where(n[..., 2:3] > 0, n.new_tensor(-1.0), n.new_tensor(1.0))
    return n * flip


@torch.no_grad()
def normal_angle_errors(
    normal_pred_cam: Tensor,  # [V, H, W, 3] unit
    normal_gt_cam: Tensor,    # [V, H, W, 3] unit
    valid: Tensor,            # [V, H, W] bool
):
    """Per-pixel angular errors (degrees) over valid pixels.

    Returns a dict with 1-D tensors ``oriented`` and ``unoriented`` (degrees).
    """
    n_pred = normal_pred_cam[valid]
    n_gt = normal_gt_cam[valid]
    if n_pred.numel() == 0:
        empty = normal_pred_cam.new_zeros((0,))
        return {"oriented": empty, "unoriented": empty}

    cos = (n_pred * n_gt).sum(dim=-1).clamp(-1.0, 1.0)
    unoriented = torch.rad2deg(torch.acos(cos.abs()))

    cos_or = (_face_camera(n_pred) * _face_camera(n_gt)).sum(dim=-1).clamp(-1.0, 1.0)
    oriented = torch.rad2deg(torch.acos(cos_or))
    return {"oriented": oriented, "unoriented": unoriented}
