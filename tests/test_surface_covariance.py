"""Surface-aligned covariance, residual reorientation, and baseline invariants."""

import math

import pytest
import torch

import torch.nn.functional as F

import src.model.encoder.common.gaussian_adapter as adapter_module
from src.model.encoder.common.gaussian_adapter import (
    GaussianAdapterCfg,
    SurfelAdapter,
    build_quaternion_from_rotation,
    pts2covariance,
)
from src.model.encoder.common.gaussians import quaternion_to_matrix

B, V, H, W = 1, 1, 6, 6
R = H * W
SH_DEGREE = 2
D_SH = (SH_DEGREE + 1) ** 2
SCALE_MIN = 1e-10
SCALE_MAX = 3.0


def _rot_z(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64).float()


def _extrinsics(rot: torch.Tensor) -> torch.Tensor:
    e = torch.eye(4)
    e[:3, :3] = rot
    e[:3, 3] = torch.tensor([0.1, -0.2, 0.3])
    return e.view(1, 1, 1, 1, 1, 4, 4).expand(B, V, 1, 1, 1, 4, 4).contiguous()


def _inputs(rot: torch.Tensor):
    k = torch.tensor([[0.7, 0.0, 0.5], [0.0, 0.7, 0.5], [0.0, 0.0, 1.0]])
    intr = k.view(1, 1, 1, 1, 1, 3, 3).expand(B, V, 1, 1, 1, 3, 3).contiguous()
    extr = _extrinsics(rot)
    ys, xs = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    coords = torch.stack([xs, ys], dim=-1).reshape(1, 1, R, 1, 1, 2).expand(B, V, R, 1, 1, 2).contiguous()
    # tilted, slightly curved depth so the analytic frame is non-trivial
    depths = (1.0 + 0.3 * xs + 0.2 * ys + 0.1 * xs * ys).reshape(1, 1, R, 1, 1).expand(B, V, R, 1, 1).contiguous()
    opac = torch.full((B, V, R, 1, 1), 0.5)
    imgs = torch.rand(B, V, 3, H, W)
    return extr, intr, coords, depths, opac, imgs


def _cfg(camera_space_frame=False, use_rotation_residual=False, **kw):
    return GaussianAdapterCfg(
        gaussian_scale_min=SCALE_MIN,
        gaussian_scale_max=SCALE_MAX,
        sh_degree=SH_DEGREE,
        camera_space_frame=camera_space_frame,
        use_rotation_residual=use_rotation_residual,
        **kw,
    )


def _base_raw(seed=7):
    torch.manual_seed(seed)
    return torch.randn(B, V, R, 1, 1, 2 + 3 * D_SH)  # scales_mul(2) + sh(3*d_sh)


def _with_delta_q(base, delta_q):
    return torch.cat([base[..., :2], delta_q, base[..., 2:]], dim=-1)


def _run(cfg, raw, rot=None):
    rot = _rot_z(0.0) if rot is None else rot
    adapter = SurfelAdapter(cfg)
    extr, intr, coords, depths, opac, imgs = _inputs(rot)
    return adapter.forward(
        extr, intr, coords, depths, opac, raw, (H, W),
        input_images=imgs,
    )


# ---------------------------------------------------------------------------
# residual identity invariant
# ---------------------------------------------------------------------------
def test_identity_delta_q_matches_analytic_covariance():
    """use_rotation_residual + Δq=[0,0,0,1] must equal the analytic-only covariance exactly."""
    base = _base_raw()
    rot = _rot_z(0.5)
    g_analytic = _run(_cfg(camera_space_frame=True, use_rotation_residual=False), base, rot)
    identity_q = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(1, 1, 1, 1, 1, 4).expand(B, V, R, 1, 1, 4)
    g_id = _run(_cfg(camera_space_frame=True, use_rotation_residual=True), _with_delta_q(base, identity_q), rot)
    assert torch.allclose(g_analytic.covariances, g_id.covariances, atol=1e-6)
    # Always-on diagnostics (scale/opacity/frame health) are now populated even for the
    # analytic-only baseline, but the Δq non-collapse key only appears with the rotation residual.
    assert "delta_q_angle" not in g_analytic.diagnostics
    assert g_id.diagnostics is not None
    assert g_id.diagnostics["delta_q_angle"].item() == pytest.approx(0.0, abs=1e-6)


def test_random_delta_q_changes_covariance():
    base = _base_raw()
    torch.manual_seed(11)
    dq = torch.randn(B, V, R, 1, 1, 4)
    g_analytic = _run(_cfg(camera_space_frame=True, use_rotation_residual=False), base)
    g_rnd = _run(_cfg(camera_space_frame=True, use_rotation_residual=True), _with_delta_q(base, dq))
    assert not torch.allclose(g_analytic.covariances, g_rnd.covariances, atol=1e-5)
    assert g_rnd.diagnostics["delta_q_angle"].item() > 0.0


def test_rotation_residual_composes_after_analytic_frame(monkeypatch):
    """A known prior and residual compose as R_a @ R_delta (Eq. 5)."""
    analytic_frame = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])  # 90 degrees about z.
    residual_frame = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])  # 90 degrees about x; these rotations do not commute.

    def known_prior(points, tangent_frame, eps):
        shape = points.shape[:-1]
        return (
            analytic_frame.expand(*shape, 3, 3),
            points.new_tensor([0.03, 0.06]).expand(*shape, 2),
            analytic_frame[:, 2].expand(*shape, 3),
        )

    monkeypatch.setattr(adapter_module, "pts2covariance", known_prior)
    delta_q = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    delta_q = delta_q.expand(B, V, R, 1, 1, 4)
    result = _run(
        _cfg(camera_space_frame=True, tangent_frame=True, use_rotation_residual=True),
        _with_delta_q(_base_raw(), delta_q),
    )
    rotation = quaternion_to_matrix(result.rotations)
    expected = (analytic_frame @ residual_frame).expand_as(rotation)
    reversed_order = (residual_frame @ analytic_frame).expand_as(rotation)
    torch.testing.assert_close(rotation, expected, rtol=0, atol=1e-6)
    assert not torch.allclose(rotation, reversed_order, rtol=0, atol=1e-6)


def test_delta_q_normalization_preserves_rotation_covariance_and_angle():
    """Quaternion magnitude changes neither reorientation nor its angle diagnostic."""
    base = _base_raw()
    cfg = _cfg(camera_space_frame=True, tangent_frame=True, use_rotation_residual=True)
    unit_q = torch.tensor([0.6, 0.0, 0.0, -0.8])
    delta_q = unit_q.expand(B, V, R, 1, 1, 4)
    unit = _run(cfg, _with_delta_q(base, delta_q))
    scaled = _run(cfg, _with_delta_q(base, 3.0 * delta_q))
    torch.testing.assert_close(scaled.covariances, unit.covariances, rtol=0, atol=1e-6)
    torch.testing.assert_close(scaled.rotations, unit.rotations, rtol=0, atol=1e-6)
    # Matrix conversion itself is scale-invariant; the diagnostic must also use
    # the normalized quaternion, including the absolute value of its w component.
    expected_angle = 2.0 * math.acos(abs(unit_q[3].item()))
    assert unit.diagnostics["delta_q_angle"].item() == pytest.approx(expected_angle, abs=1e-6)
    assert scaled.diagnostics["delta_q_angle"].item() == pytest.approx(expected_angle, abs=1e-6)


# ---------------------------------------------------------------------------
# analytic coordinate consistency: c2w applied exactly once
# ---------------------------------------------------------------------------
def test_camera_space_frame_applies_c2w_once():
    """Camera-space covariance is extrinsic-independent, so cov(R) == R @ cov(I) @ R^T."""
    base = _base_raw()
    rot = _rot_z(0.6)
    cfg = _cfg(camera_space_frame=True)
    g_i = _run(cfg, base, _rot_z(0.0))
    g_r = _run(cfg, base, rot)
    expected = rot @ g_i.covariances @ rot.transpose(-1, -2)
    assert torch.allclose(g_r.covariances, expected, atol=1e-5)


def test_world_space_frame_double_rotates():
    """Regression guard: the inherited world-space baseline does not satisfy the single-rotation law."""
    base = _base_raw()
    rot = _rot_z(0.6)
    cfg = _cfg(camera_space_frame=False)
    g_i = _run(cfg, base, _rot_z(0.0))
    g_r = _run(cfg, base, rot)
    expected = rot @ g_i.covariances @ rot.transpose(-1, -2)
    assert not torch.allclose(g_r.covariances, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Covariance numerics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("camera_space_frame", [False, True])
@pytest.mark.parametrize("use_rotation_residual", [False, True])
def test_covariance_symmetric_and_psd(camera_space_frame, use_rotation_residual):
    base = _base_raw()
    if use_rotation_residual:
        torch.manual_seed(3)
        raw = _with_delta_q(base, torch.randn(B, V, R, 1, 1, 4))
    else:
        raw = base
    g = _run(_cfg(camera_space_frame, use_rotation_residual), raw, _rot_z(0.4))
    cov = g.covariances.reshape(-1, 3, 3)
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6)
    # surfel covariance is rank-2 (scale_min ~ 0): eigenvalues must be >= small negative tol
    eig = torch.linalg.eigvalsh(cov)
    assert eig.min().item() > -1e-6


def test_normals_unit_norm_with_residual():
    base = _base_raw()
    torch.manual_seed(5)
    g = _run(_cfg(camera_space_frame=True, use_rotation_residual=True),
             _with_delta_q(base, torch.randn(B, V, R, 1, 1, 4)))
    assert torch.allclose(g.normals.norm(dim=-1), torch.ones(B, V, R, 1, 1), atol=1e-5)


# ---------------------------------------------------------------------------
# Quaternion convention
# ---------------------------------------------------------------------------
def test_quaternion_to_matrix_identity():
    q = torch.tensor([0.0, 0.0, 0.0, 1.0])  # w-last identity
    assert torch.allclose(quaternion_to_matrix(q), torch.eye(3), atol=1e-6)


def test_quaternion_round_trip():
    """build_quaternion_from_rotation(quaternion_to_matrix(q)) == ±q (w-last)."""
    torch.manual_seed(0)
    q = torch.randn(8, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    rot = quaternion_to_matrix(q)
    q_rt = build_quaternion_from_rotation(rot)  # returns [x, y, z, w]
    q_rt = q_rt / q_rt.norm(dim=-1, keepdim=True)
    dots = (q * q_rt).sum(dim=-1).abs()  # sign-agnostic
    assert torch.allclose(dots, torch.ones(8), atol=1e-4)


def test_analytic_normal_aligns_with_third_column_approx():
    """pts2covariance's 3rd column ~ (±) the surface normal.

    NOTE: SurfSplat's ``build_gs_rotation_from_normal`` is an *approximate*
    (non-orthonormal) Rodrigues construction, so the alignment is close but not exact.
    We preserve that baseline behaviour and only assert approximate agreement.
    """
    torch.manual_seed(1)
    pts = torch.randn(1, 1, H, W, 3)
    rotations, _, normals = pts2covariance(pts)
    third_col = torch.nn.functional.normalize(rotations[..., :, 2], dim=-1)
    cos = (third_col * normals).sum(dim=-1).abs()  # frame normal may be sign-flipped (camera-facing)
    assert cos.mean().item() > 0.9


# ---------------------------------------------------------------------------
# tangent_frame tangent-aligned orthonormal frame
# ---------------------------------------------------------------------------
def _tilted_plane_points(x_step, y_step):
    """Camera-space point map of a tilted plane: X(i,j) = j·x_step + i·y_step.

    Sobel central differences then recover ∂X/∂x = x_step and ∂X/∂y = y_step exactly, so the
    analytic tangent frame is known in closed form.
    """
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    pts = xs[..., None].float() * x_step + ys[..., None].float() * y_step  # [H, W, 3]
    return pts.view(1, 1, H, W, 3)


def test_tangent_frame_is_orthonormal_rotation():
    """tangent_frame frame must be a proper rotation: RᵀR = I and det(R) = +1, even on random input."""
    torch.manual_seed(2)
    pts = torch.randn(1, 1, H, W, 3)
    rotations, _, _ = pts2covariance(pts, tangent_frame=True)
    R = rotations.reshape(-1, 3, 3)
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R.transpose(-1, -2) @ R, eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(R.shape[0]), atol=1e-5)


def test_tangent_frame_axes_match_analytic_plane():
    """On a known tilted plane, column 2 (n) ∥ plane normal, column 0 (e1) ∥ ∂X/∂x, and the
    returned normal equals column 2 (camera-facing, +z)."""
    x_step = torch.tensor([1.0, 0.0, 0.3])
    y_step = torch.tensor([0.0, 1.0, 0.2])
    pts = _tilted_plane_points(x_step, y_step)
    rotations, _, normals = pts2covariance(pts, tangent_frame=True)
    R = rotations.reshape(-1, 3, 3)

    n_an = F.normalize(torch.cross(x_step, y_step, dim=0), dim=0)
    n_an = n_an * torch.sign(n_an[2])  # camera-facing +z
    e1_an = F.normalize(x_step - (x_step @ n_an) * n_an, dim=0)

    col_n = R[:, :, 2]
    col_e1 = R[:, :, 0]
    assert (col_n * n_an).sum(-1).abs().mean().item() > 0.999       # thin axis ∥ plane normal
    assert (col_e1 * e1_an).sum(-1).abs().mean().item() > 0.999     # e1 ∥ image-x tangent
    assert col_n[:, 2].min().item() > 0.0                           # camera-facing convention
    # returned normals == frame third column
    assert torch.allclose(normals.reshape(-1, 3), col_n, atol=1e-5)


def test_tangent_frame_normal_matches_baseline_direction():
    """tangent_frame preserves the thin axis: its normal matches the baseline (normal-only) frame's
    cross-product normal up to sign, so a tangent_frame-vs-analytic ablation isolates in-plane alignment."""
    torch.manual_seed(4)
    pts = torch.randn(1, 1, H, W, 3)
    _, _, n_tangent_frame = pts2covariance(pts, tangent_frame=True)
    _, _, n_base = pts2covariance(pts, tangent_frame=False)
    cos = (F.normalize(n_tangent_frame, dim=-1) * F.normalize(n_base, dim=-1)).sum(-1).abs()
    assert cos.mean().item() > 0.99


def test_tangent_frame_default_is_off_and_changes_covariance():
    """Default path is byte-identical to the baseline; the flag changes the frame."""
    torch.manual_seed(6)
    pts = torch.randn(1, 1, H, W, 3)
    r_default, s_default, n_default = pts2covariance(pts)
    r_off, s_off, n_off = pts2covariance(pts, tangent_frame=False)
    assert torch.equal(r_default, r_off) and torch.equal(s_default, s_off)
    r_on, _, _ = pts2covariance(pts, tangent_frame=True)
    assert not torch.allclose(r_default, r_on, atol=1e-4)


def test_tangent_frame_config_default_off():
    assert SurfelAdapter(_cfg()).cfg.tangent_frame is False


@pytest.mark.parametrize("use_rotation_residual", [False, True])
def test_tangent_frame_adapter_covariance_spd(use_rotation_residual):
    """Full adapter forward with tangent_frame (optionally + Δq) yields symmetric, PSD covariance."""
    base = _base_raw()
    if use_rotation_residual:
        torch.manual_seed(8)
        raw = _with_delta_q(base, torch.randn(B, V, R, 1, 1, 4))
    else:
        raw = base
    g = _run(_cfg(camera_space_frame=True, tangent_frame=True,
                  use_rotation_residual=use_rotation_residual), raw, _rot_z(0.4))
    cov = g.covariances.reshape(-1, 3, 3)
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6)
    assert torch.linalg.eigvalsh(cov).min().item() > -1e-6
    assert torch.isfinite(cov).all()


def test_tangent_frame_thin_axis_aligns_with_normal_via_adapter():
    """The smallest-eigenvalue eigenvector of the tangent_frame surfel covariance must align with the
    rendered surface normal (covariance is thin along n) — surface alignment."""
    base = _base_raw()
    g = _run(_cfg(camera_space_frame=True, tangent_frame=True), base, _rot_z(0.3))
    cov = g.covariances.reshape(-1, 3, 3)
    n = F.normalize(g.normals.reshape(-1, 3), dim=-1)
    _, evecs = torch.linalg.eigh(cov)        # ascending eigenvalues
    v_min = evecs[:, :, 0]                    # thinnest axis
    assert (v_min * n).sum(-1).abs().mean().item() > 0.99


def test_tangent_frame_colinear_tangents_stay_proper_rotation():
    """Colinear tangents use the optical-axis fallback to keep a proper rotation."""
    d = torch.tensor([0.3, 0.4, 1.0])
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    pts = ((xs + ys).float()[..., None] * d).view(1, 1, H, W, 3)  # X=(x+y)·d ⇒ t_u=t_v=d (colinear)
    rotations, scales, normals = pts2covariance(pts, tangent_frame=True)
    R = rotations.reshape(-1, 3, 3)
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R.transpose(-1, -2) @ R, eye, atol=1e-5)                    # orthonormal
    assert torch.allclose(torch.linalg.det(R), torch.ones(R.shape[0]), atol=1e-5)     # det +1 (not 0)
    assert torch.allclose(normals.reshape(-1, 3).norm(dim=-1), torch.ones(R.shape[0]), atol=1e-5)
    assert (scales[..., 0].reshape(-1) > 0).all()                                     # s1 = ‖t_u‖ > 0


def test_identity_delta_q_matches_tangent_frame_covariance():
    """Under the tangent_frame frame, Δq = identity must reduce to the tangent_frame-only covariance exactly (residual
    identity invariant on the new frame — mirrors the analytic test)."""
    base = _base_raw()
    rot = _rot_z(0.5)
    g_tangent_frame = _run(_cfg(camera_space_frame=True, tangent_frame=True, use_rotation_residual=False), base, rot)
    identity_q = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(1, 1, 1, 1, 1, 4).expand(B, V, R, 1, 1, 4)
    g_id = _run(_cfg(camera_space_frame=True, tangent_frame=True, use_rotation_residual=True),
                _with_delta_q(base, identity_q), rot)
    assert torch.allclose(g_tangent_frame.covariances, g_id.covariances, atol=1e-6)
    assert g_id.diagnostics["delta_q_angle"].item() == pytest.approx(0.0, abs=1e-6)


def test_tangent_frame_changes_inplane_covariance_but_preserves_normal_vs_analytic():
    """Through the adapter: switching the frame from analytic (normal-only) to
    tangent_frame (tangent-aligned) changes the in-plane covariance while keeping the thin-axis
    normal direction — so tangent_frame vs analytic isolates in-plane tangent alignment, not a relabeling."""
    base = _base_raw()
    rot = _rot_z(0.4)
    g_analytic = _run(_cfg(camera_space_frame=True, tangent_frame=False), base, rot)
    g_tangent_frame = _run(_cfg(camera_space_frame=True, tangent_frame=True), base, rot)
    assert not torch.allclose(g_analytic.covariances, g_tangent_frame.covariances, atol=1e-4)   # covariance differs
    cos = (F.normalize(g_analytic.normals, dim=-1) * F.normalize(g_tangent_frame.normals, dim=-1)).sum(-1).abs()
    assert cos.mean().item() > 0.9                                              # normal direction kept


# ---------------------------------------------------------------------------
# d_in / channel layout
# ---------------------------------------------------------------------------
def test_d_in_grows_by_four_with_residual():
    assert SurfelAdapter(_cfg(use_rotation_residual=False)).d_in == 2 + 3 * D_SH
    assert SurfelAdapter(_cfg(use_rotation_residual=True)).d_in == 2 + 4 + 3 * D_SH
