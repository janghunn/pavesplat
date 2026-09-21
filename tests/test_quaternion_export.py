"""Quaternion export: finite rotations, round trips, and w-last unit conventions."""

import torch

from src.model.encoder.common.gaussian_adapter import build_quaternion_from_rotation
from src.model.encoder.common.gaussians import quaternion_to_matrix


def _random_unit_quats(n, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(n, 4, dtype=torch.float64).float()
    return q / q.norm(dim=-1, keepdim=True)


def _naive(rot):
    """The trace-pivot formula (valid only when w is bounded away from 0)."""
    w = torch.sqrt(1 + rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]) / 2
    x = (rot[:, 2, 1] - rot[:, 1, 2]) / (4 * w)
    y = (rot[:, 0, 2] - rot[:, 2, 0]) / (4 * w)
    z = (rot[:, 1, 0] - rot[:, 0, 1]) / (4 * w)
    return torch.stack((x, y, z, w), dim=1)


def test_round_trip_random_rotations():
    q = _random_unit_quats(512)
    R = quaternion_to_matrix(q)
    q2 = build_quaternion_from_rotation(R)
    R2 = quaternion_to_matrix(q2)
    assert torch.isfinite(q2).all()
    assert torch.allclose(R2, R, atol=1e-5)


def test_output_conventions_unit_norm_and_w_nonnegative():
    q = _random_unit_quats(256, seed=1)
    q2 = build_quaternion_from_rotation(quaternion_to_matrix(q))
    assert torch.allclose(q2.norm(dim=-1), torch.ones(q2.shape[0]), atol=1e-5)
    assert (q2[:, 3] >= 0).all()  # w-last, double cover resolved to w >= 0


def test_identity_maps_to_identity_quaternion():
    R = torch.eye(3).unsqueeze(0)
    q = build_quaternion_from_rotation(R)
    assert torch.allclose(q, torch.tensor([[0.0, 0.0, 0.0, 1.0]]), atol=1e-6)


def test_180_degree_rotations_finite_and_exact():
    # w = 0 exactly: the naive formula divides by zero here. Axis-aligned + random axes.
    torch.manual_seed(2)
    axes = torch.cat([torch.eye(3), torch.nn.functional.normalize(torch.randn(8, 3), dim=-1)])
    q180 = torch.cat([axes, torch.zeros(axes.shape[0], 1)], dim=-1)  # w-last, w=0
    R = quaternion_to_matrix(q180)
    q2 = build_quaternion_from_rotation(R)
    assert torch.isfinite(q2).all()
    assert torch.allclose(quaternion_to_matrix(q2), R, atol=1e-5)
    # Regression guard: the naive form really is non-finite on these inputs.
    assert not torch.isfinite(_naive(R)).all()


def test_tangent_frame_edge_case_trace_minus_one():
    # R = [e1, e2, n] columns with e1=-x, e2=-y, n=+z (trace -1): a camera-facing tangent
    # frame reachable at depth edges.
    R = torch.diag(torch.tensor([-1.0, -1.0, 1.0])).unsqueeze(0)
    q = build_quaternion_from_rotation(R)
    assert torch.isfinite(q).all()
    assert torch.allclose(quaternion_to_matrix(q), R, atol=1e-6)
    # 180 degrees about z, w-last, w >= 0 convention.
    assert torch.allclose(q.abs(), torch.tensor([[0.0, 0.0, 1.0, 0.0]]), atol=1e-6)


def test_matches_naive_formula_away_from_singularity():
    # The pivot conversion agrees with the trace formula away from w = 0.
    torch.manual_seed(3)
    q = _random_unit_quats(256, seed=3)
    q = torch.where(q[:, 3:4] < 0, -q, q)        # w >= 0
    q = q[q[:, 3] > 0.3]                          # away from the w ~ 0 singular band
    R = quaternion_to_matrix(q)
    q_new = build_quaternion_from_rotation(R)
    q_old = _naive(R)
    assert torch.allclose(q_new, q_old, atol=1e-4)
