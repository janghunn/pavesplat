"""Paper metric regressions with hand-computed RGB and accumulated-alpha values."""

from types import SimpleNamespace

import pytest
import torch

from src.model.model_wrapper import ModelWrapper


def _stub():
    return SimpleNamespace(
        test_cfg=SimpleNamespace(
            eval_coverage=True, render_chunk_size=None,
            black_eps_sweep=[0.01, 0.02, 0.05],
        ),
        _coverage_taus=[0.01, 0.05, 0.1, 0.5],
        _cov_tau_key=ModelWrapper._cov_tau_key,
        test_step_outputs={},
    )


@pytest.mark.parametrize("scale", [1, 2, 4, 8])
def test_black_hole_strict_boundaries_gt_mask_and_sweep(scale):
    # Columns: a hole; GT below/at epsilon; render at epsilon; a single bright
    # channel; a 0.03 render; another bright GT; both channels at zero.
    rendered = torch.zeros(2, 3, 1, 4)
    gt = torch.zeros_like(rendered)
    rendered[:, 1] = torch.tensor([[[0, 0, 0, 0.02]], [[0.03, 0.03, 0.009, 0]]])
    gt[:, 2] = torch.tensor([[[0.1, 0.01, 0.02, 0.1]], [[0.1, 0.04, 0.03, 0]]])
    stub = _stub()
    ModelWrapper._append_black_frac(stub, rendered, gt, scale)
    assert stub.test_step_outputs == {
        f"cov_black_s{scale}": pytest.approx([2 / 8]),
        f"cov_blackany_s{scale}": pytest.approx([5 / 8]),
        f"cov_black_s{scale}_eps01": pytest.approx([3 / 8]),
        f"cov_black_s{scale}_eps02": pytest.approx([2 / 8]),
        f"cov_black_s{scale}_eps05": pytest.approx([3 / 8]),
    }
    # A configured sweep must be read directly, without silently using defaults.
    stub.test_cfg.black_eps_sweep = [0.03]
    stub.test_step_outputs.clear()
    ModelWrapper._append_black_frac(stub, rendered, gt, scale)
    assert stub.test_step_outputs[f"cov_black_s{scale}_eps03"] == pytest.approx([2 / 8])
    assert f"cov_black_s{scale}_eps01" not in stub.test_step_outputs


@pytest.mark.parametrize("scale", [1, 2, 4, 8])
def test_alpha_coverage_every_threshold_and_inclusive_boundary(scale):
    stub = _stub()
    alpha = torch.tensor([[[0.0, 0.01, 0.049, 0.05]], [[0.099, 0.1, 0.5, 1.0]]])
    ModelWrapper._append_coverage(stub, alpha, scale)
    assert stub.test_step_outputs == {
        f"cov_meanalpha_s{scale}": pytest.approx([1.808 / 8]),
        f"cov_C01_s{scale}": pytest.approx([7 / 8]),
        f"cov_C05_s{scale}": pytest.approx([5 / 8]),
        f"cov_C10_s{scale}": pytest.approx([3 / 8]),
        f"cov_C50_s{scale}": pytest.approx([2 / 8]),
    }


@pytest.mark.parametrize("disabled,chunk_size", [(True, None), (False, 1)])
def test_disabled_or_chunked_eval_emits_no_coverage(disabled, chunk_size):
    stub = _stub()
    stub.test_cfg.eval_coverage = not disabled
    stub.test_cfg.render_chunk_size = chunk_size
    ModelWrapper._append_coverage(stub, torch.ones(2, 1, 4), 1)
    ModelWrapper._append_black_frac(stub, torch.zeros(2, 3, 1, 4), torch.ones(2, 3, 1, 4), 1)
    assert stub.test_step_outputs == {}
