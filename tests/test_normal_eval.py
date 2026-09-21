"""Rendered-normal angular errors, GT depth normals, and camera-frame alignment."""

import numpy as np
import torch

from src.evaluation.normal_eval import (
    gt_normal_from_depth,
    normal_angle_errors,
    rendered_normal_to_camera,
)


K = torch.tensor([[0.7, 0.0, 0.5], [0.0, 0.7, 0.5], [0.0, 0.0, 1.0]])


def test_fronto_parallel_plane_normal_is_camera_axis():
    # Constant-depth plane => camera-space normal = (0, 0, ±1).
    V, H, W = 1, 24, 24
    depth = torch.full((V, H, W), 3.0)
    n_cam, valid = gt_normal_from_depth(depth, K.view(1, 3, 3), edge_rel_thresh=0.05)
    inner = n_cam[valid]
    assert inner.numel() > 0
    # |z component| ~ 1, x/y ~ 0.
    assert inner[:, 2].abs().mean() > 0.999
    assert inner[:, :2].abs().mean() < 1e-3


def test_rendered_to_camera_identity_extrinsics():
    V, H, W = 1, 8, 8
    n_world = torch.randn(V, 3, H, W)
    c2w = torch.eye(4).view(1, 4, 4)
    n_cam, cov = rendered_normal_to_camera(n_world, c2w)
    expected = torch.nn.functional.normalize(n_world.permute(0, 2, 3, 1), dim=-1)
    assert torch.allclose(n_cam, expected, atol=1e-6)
    assert (cov > 0).all()


def test_rendered_to_camera_rotation():
    # 90deg rotation about x: world->cam rotation must invert it.
    V, H, W = 1, 4, 4
    theta = torch.tensor(np.pi / 2)
    R = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, torch.cos(theta).item(), -torch.sin(theta).item()],
        [0.0, torch.sin(theta).item(), torch.cos(theta).item()],
    ])
    c2w = torch.eye(4)
    c2w[:3, :3] = R
    n_world = torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1).expand(V, 3, H, W).contiguous()
    n_cam, _ = rendered_normal_to_camera(n_world, c2w.view(1, 4, 4))
    # R_w2c = R^T; R^T @ (0,0,1) = third column of R^T = third row of R = (0,1,0).
    expected = torch.tensor([0.0, 1.0, 0.0])
    assert torch.allclose(n_cam[0, 0, 0], expected, atol=1e-5)


def test_matching_normals_zero_angle():
    # GT plane normal compared with an identical rendered normal => ~0 angle.
    V, H, W = 1, 24, 24
    depth = torch.full((V, H, W), 2.5)
    n_gt, valid = gt_normal_from_depth(depth, K.view(1, 3, 3))
    # "rendered" world normal = the same (0,0,-1) camera normal, identity extrinsics.
    n_world = torch.tensor([0.0, 0.0, -1.0]).view(1, 3, 1, 1).expand(V, 3, H, W).contiguous()
    n_pred, cov = rendered_normal_to_camera(n_world, torch.eye(4).view(1, 4, 4))
    errs = normal_angle_errors(n_pred, n_gt, valid)
    assert errs["unoriented"].numel() > 0
    assert errs["unoriented"].mean().item() < 1e-2
    assert errs["oriented"].mean().item() < 1e-2


def test_orthogonal_normals_ninety_deg():
    V, H, W = 1, 12, 12
    n_gt = torch.tensor([0.0, 0.0, -1.0]).view(1, 1, 1, 3).expand(V, H, W, 3).contiguous()
    n_pred = torch.tensor([1.0, 0.0, 0.0]).view(1, 1, 1, 3).expand(V, H, W, 3).contiguous()
    valid = torch.ones(V, H, W, dtype=torch.bool)
    errs = normal_angle_errors(n_pred, n_gt, valid)
    assert abs(errs["unoriented"].mean().item() - 90.0) < 1e-2


def test_edge_mask_excludes_depth_jump():
    # A depth step in the middle should knock out the discontinuity band.
    V, H, W = 1, 32, 32
    depth = torch.full((V, H, W), 2.0)
    depth[:, :, 16:] = 5.0  # vertical depth cliff
    _, valid = gt_normal_from_depth(depth, K.view(1, 3, 3), edge_rel_thresh=0.05)
    # Columns straddling the cliff (15,16) must be invalidated.
    assert not valid[:, 5:25, 15].any()
    assert not valid[:, 5:25, 16].any()
    # Flat interior away from the cliff stays valid.
    assert valid[:, 5:25, 5].any()
