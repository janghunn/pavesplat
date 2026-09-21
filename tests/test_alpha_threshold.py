"""CPU regressions for the train/eval alpha thresholds and compositing boundaries."""

from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

import src.model.decoder.cuda_splatting as cuda_splatting
import src.model.decoder.decoder_splatting_cuda as decoder_module
from src.model.decoder.decoder_splatting_cuda import (
    DecoderSplattingCUDA,
    DecoderSplattingCUDACfg,
)
from src.model.types import Gaussians

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------
# (a) Config defaults pinned — dataclass and yaml must both carry the thresholds
# --------------------------------------------------------------------------------------
def test_dataclass_defaults_pinned():
    cfg = DecoderSplattingCUDACfg(name="splatting_cuda")
    assert cfg.alpha_threshold_train == 0.1
    assert cfg.alpha_threshold_eval == 0.001


def test_yaml_defaults_pinned():
    with (REPO_ROOT / "config" / "model" / "decoder" / "splatting_cuda.yaml").open() as f:
        cfg = yaml.safe_load(f)
    assert cfg["alpha_threshold_train"] == 0.1
    assert cfg["alpha_threshold_eval"] == 0.001


def test_render_cuda_signature_default_is_eval_threshold():
    import inspect

    sig = inspect.signature(cuda_splatting.render_cuda)
    assert sig.parameters["alpha_threshold"].default == 0.001


# --------------------------------------------------------------------------------------
# (b) Decoder selects the threshold by self.training
# --------------------------------------------------------------------------------------
def _decoder(cfg=None):
    cfg = cfg or DecoderSplattingCUDACfg(name="splatting_cuda")
    dataset_cfg = SimpleNamespace(background_color=[0.0, 0.0, 0.0])
    return DecoderSplattingCUDA(cfg, dataset_cfg)


def _minimal_inputs(b=1, v=1, g=2, h=4, w=4):
    gaussians = Gaussians(
        means=torch.zeros(b, g, 3),
        covariances=torch.eye(3).expand(b, g, 3, 3).contiguous(),
        harmonics=torch.zeros(b, g, 3, 1),
        opacities=torch.full((b, g), 0.5),
        scales_mul=torch.ones(b, g, 2),
        normals=torch.zeros(b, g, 3),
    )
    extrinsics = torch.eye(4).expand(b, v, 4, 4).contiguous()
    intrinsics = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    intrinsics = intrinsics.expand(b, v, 3, 3).contiguous()
    near = torch.ones(b, v)
    far = torch.full((b, v), 100.0)
    return gaussians, extrinsics, intrinsics, near, far, (h, w)


def test_decoder_passes_train_then_eval_threshold(monkeypatch):
    captured = {}

    def fake_render_cuda(*args, alpha_threshold, return_alpha=False, **kwargs):
        captured["alpha_threshold"] = alpha_threshold
        color = torch.zeros(1, 3, 4, 4)  # [(b v), 3, h, w]
        return (color, torch.zeros(1, 1, 4, 4)) if return_alpha else color

    monkeypatch.setattr(decoder_module, "render_cuda", fake_render_cuda)
    decoder = _decoder()
    inputs = _minimal_inputs()

    decoder.train()
    decoder.forward(*inputs)
    assert captured["alpha_threshold"] == 0.1

    decoder.eval()
    decoder.forward(*inputs)
    assert captured["alpha_threshold"] == 0.001


def test_decoder_respects_custom_cfg_thresholds(monkeypatch):
    captured = {}

    def fake_render_cuda(*args, alpha_threshold, return_alpha=False, **kwargs):
        captured["alpha_threshold"] = alpha_threshold
        color = torch.zeros(1, 3, 4, 4)
        return (color, torch.zeros(1, 1, 4, 4)) if return_alpha else color

    monkeypatch.setattr(decoder_module, "render_cuda", fake_render_cuda)
    cfg = DecoderSplattingCUDACfg(
        name="splatting_cuda", alpha_threshold_train=0.2, alpha_threshold_eval=0.005
    )
    decoder = _decoder(cfg)
    inputs = _minimal_inputs()

    decoder.train()
    decoder.forward(*inputs)
    assert captured["alpha_threshold"] == 0.2

    decoder.eval()
    decoder.forward(*inputs)
    assert captured["alpha_threshold"] == 0.005


# --------------------------------------------------------------------------------------
# (c) render_cuda where-branch semantics via a fake rasterizer (CPU, no CUDA)
# --------------------------------------------------------------------------------------
class _FakeRasterizer:
    """Stands in for diff_gaussian_rasterization.GaussianRasterizer: returns a fixed
    premultiplied image whose pixel values EQUAL alpha (i.e. solid color 1.0 under
    premultiplied alpha), so the un-premultiply branch is trivially checkable."""

    alpha = torch.tensor([[0.5, 0.05], [0.0005, 0.9]])

    def __init__(self, settings):
        pass

    def __call__(self, **kwargs):
        image = self.alpha.expand(3, 2, 2).clone()
        radii = torch.zeros(kwargs["means3D"].shape[0])
        depth = torch.ones(1, 2, 2)
        return image, radii, depth, self.alpha.unsqueeze(0).clone()


def _render(monkeypatch, alpha_threshold, rasterizer=None):
    monkeypatch.setattr(cuda_splatting, "GaussianRasterizer", rasterizer or _FakeRasterizer)
    b, g, h, w = 1, 2, 2, 2
    return cuda_splatting.render_cuda(
        torch.eye(4).expand(b, 4, 4).contiguous(),
        torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]),
        torch.ones(b),
        torch.full((b,), 100.0),
        (h, w),
        torch.zeros(b, 3),
        torch.zeros(b, g, 3),
        torch.eye(3).expand(b, g, 3, 3).contiguous(),
        torch.zeros(b, g, 3, 1),
        torch.full((b, g), 0.5),
        use_sh=False,
        alpha_threshold=alpha_threshold,
    )


def test_where_branch_train_threshold_keeps_premultiplied_dark(monkeypatch):
    out = _render(monkeypatch, alpha_threshold=0.1)[0]  # [3, 2, 2]
    # alpha > 0.1 -> un-premultiplied to the solid color 1.0
    assert torch.allclose(out[:, 0, 0], torch.ones(3))
    assert torch.allclose(out[:, 1, 1], torch.ones(3))
    # alpha <= 0.1 -> stays premultiplied (dark on black bg => photometric loss bites):
    # Under the eval threshold these pixels would be divided.
    assert torch.allclose(out[:, 0, 1], torch.full((3,), 0.05))
    assert torch.allclose(out[:, 1, 0], torch.full((3,), 0.0005))


def test_where_branch_eval_threshold_normalizes_more_pixels(monkeypatch):
    out = _render(monkeypatch, alpha_threshold=0.001)[0]
    # alpha > 0.001 -> normalized (including the 0.05 pixel the train threshold keeps dark)
    for ij in [(0, 0), (0, 1), (1, 1)]:
        assert torch.allclose(out[:, ij[0], ij[1]], torch.ones(3))
    # alpha = 0.0005 <= 0.001 -> stays premultiplied
    assert torch.allclose(out[:, 1, 0], torch.full((3,), 0.0005))
    # the clamp_min(1e-8) guard: no inf/NaN anywhere even with tiny alphas
    assert torch.isfinite(out).all()


class _FakeRasterizerBoundary(_FakeRasterizer):
    # Pixels sitting EXACTLY on the two thresholds, plus one clearly above.
    alpha = torch.tensor([[0.1, 0.001], [0.5, 0.9]])


def test_where_branch_is_strict_greater_than(monkeypatch):
    # Compositing uses strict `>`: a pixel at exactly alpha == threshold stays premultiplied.
    # A silent flip to `>=` would alter training numerics undetected otherwise.
    out_train = _render(monkeypatch, alpha_threshold=0.1, rasterizer=_FakeRasterizerBoundary)[0]
    assert torch.allclose(out_train[:, 0, 0], torch.full((3,), 0.1))   # == 0.1 stays
    out_eval = _render(monkeypatch, alpha_threshold=0.001, rasterizer=_FakeRasterizerBoundary)[0]
    assert torch.allclose(out_eval[:, 0, 1], torch.full((3,), 0.001))  # == 0.001 stays
    assert torch.allclose(out_eval[:, 0, 0], torch.ones(3))            # > 0.001 normalized


# --------------------------------------------------------------------------------------
# (d) render_cuda_orthographic (figure path) — same semantics, default 0.001
# --------------------------------------------------------------------------------------
def test_orthographic_signature_default_is_eval_threshold():
    import inspect

    sig = inspect.signature(cuda_splatting.render_cuda_orthographic)
    assert sig.parameters["alpha_threshold"].default == 0.001


def _render_orthographic(monkeypatch, **kw):
    monkeypatch.setattr(cuda_splatting, "GaussianRasterizer", _FakeRasterizer)
    b, g, h, w = 1, 2, 2, 2
    return cuda_splatting.render_cuda_orthographic(
        torch.eye(4).expand(b, 4, 4).contiguous(),
        torch.ones(b),                 # width
        torch.ones(b),                 # height
        torch.ones(b),
        torch.full((b,), 100.0),
        (h, w),
        torch.zeros(b, 3),
        torch.zeros(b, g, 3),
        torch.eye(3).expand(b, g, 3, 3).contiguous(),
        torch.zeros(b, g, 3, 1),
        torch.full((b, g), 0.5),
        use_sh=False,
        **kw,
    )


def test_orthographic_where_branch_and_clamp(monkeypatch):
    out = _render_orthographic(monkeypatch)[0]  # default 0.001
    assert torch.allclose(out[:, 0, 0], torch.ones(3))                # 0.5 normalized
    assert torch.allclose(out[:, 0, 1], torch.ones(3))                # 0.05 normalized
    assert torch.allclose(out[:, 1, 0], torch.full((3,), 0.0005))     # below -> premultiplied
    assert torch.isfinite(out).all()                                  # clamp_min guard
    out_high = _render_orthographic(monkeypatch, alpha_threshold=0.1)[0]
    assert torch.allclose(out_high[:, 0, 1], torch.full((3,), 0.05))  # 0.05 <= 0.1 stays
