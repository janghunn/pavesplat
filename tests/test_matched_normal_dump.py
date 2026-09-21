"""CPU integration tests for the matched-pixel normal archives and manifest."""

from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

import src.model.model_wrapper as wrapper_module
from src.model.model_wrapper import ModelWrapper, TestCfg as EvaluationCfg


@pytest.fixture
def fake_evaluation(tmp_path, monkeypatch):
    height = width = 8
    intrinsics = torch.tensor([[0.7, 0, 0.5], [0, 0.7, 0.5], [0, 0, 1.0]])
    target = {
        "image": torch.ones(1, 2, 3, height, width),
        "depth": torch.full((1, 2, height, width), 3.0),
        "intrinsics": intrinsics.expand(1, 2, 3, 3).contiguous(),
        "extrinsics": torch.eye(4).expand(1, 2, 4, 4).contiguous(),
        "near": torch.full((1, 2), 0.5),
        "far": torch.full((1, 2), 20.0),
        "index": torch.tensor([[7, 19]]),
    }
    batch = {"scene": ["scene00"], "target": target, "context": target}
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"scene00": {"target": [7, 19]}}))
    cfg = {
        "output_dir": str(tmp_path),
        "checkpointing": {"load": "model.ckpt"},
        "dataset": {"view_sampler": {"index_path": str(index_path)}},
        "experiment": "zeroshot_scannet_pavesplat_vitl",
    }
    monkeypatch.setattr(wrapper_module, "get_cfg", lambda: cfg)
    for name in ("compute_psnr", "compute_ssim", "compute_lpips"):
        monkeypatch.setattr(wrapper_module, name, lambda *args: torch.tensor([1.0]))

    def render(*args, **kwargs):
        h, w = args[5]
        normal = torch.zeros(1, 2, 3, h, w)
        normal[0, 0, 2] = -1
        normal[0, 1, 0] = 1
        normal[0, 0, :, 2, 2] = 0  # A missing rendered hit with valid GT depth.
        return SimpleNamespace(
            color=torch.ones(1, 2, 3, h, w),
            depth=torch.full((1, 2, h, w), 3.0),
            normal=normal, alpha=None,
        )

    test_cfg = EvaluationCfg(
        output_path=tmp_path, compute_scores=True,
        save_image=False, save_video=False, eval_time_skip_steps=0,
        save_gt_image=False, save_input_images=False, save_depth=False,
        save_depth_concat_img=False, save_depth_npy=False, save_gaussian=False,
        render_chunk_size=None, stablize_camera=False, stab_camera_kernel=1,
        eval_normal_error=True, save_normal_npz=True, hrrc_eval_8x_only=True,
    )
    stub = SimpleNamespace(
        test_cfg=test_cfg,
        train_cfg=SimpleNamespace(forward_depth_only=False, hrrc_gt_mode="bicubic"),
        encoder=lambda *args, **kwargs: object(),
        decoder=SimpleNamespace(forward=render),
        benchmarker=SimpleNamespace(time=lambda *args, **kwargs: nullcontext()),
        global_step=0, data_shim=lambda batch: batch,
        test_step_outputs={}, time_skip_steps_dict={"encoder": 0, "decoder": 0},
        _normal_buf={"oriented": [], "unoriented": []}, _normal_npz_scenes=[],
        _alpha_view=ModelWrapper._alpha_view,
    )
    for name in ("_append_coverage", "_append_black_frac"):
        setattr(stub, name, MethodType(getattr(ModelWrapper, name), stub))
    return stub, batch, cfg


@pytest.mark.parametrize("second_depth", [4.0, 10.0])
def test_two_view_dump_keys_shapes_values_and_manifest(fake_evaluation, tmp_path, second_depth):
    stub, batch, cfg = fake_evaluation
    batch["target"]["depth"][0, 1] = second_depth
    ModelWrapper.test_step(stub, batch, 0)
    ModelWrapper._write_normal_npz_manifest(stub)
    with np.load(tmp_path / "normal_npz" / "scene00.npz") as dump:
        assert set(dump.files) == {
            "target_index", "n_pred", "n_gt", "unoriented", "oriented", "valid", "valid_gt_range",
        }
        np.testing.assert_array_equal(dump["target_index"], [7, 19])
        assert dump["target_index"].dtype == np.int32
        expected_gt_range = np.zeros((2, 8, 8), dtype=bool)
        expected_gt_range[0, 1:-1, 1:-1] = True
        expected_gt_range[1, 1:-1, 1:-1] = second_depth < 10.0
        expected_valid = expected_gt_range.copy()
        expected_valid[0, 2, 2] = False
        for key, expected in (("valid_gt_range", expected_gt_range), ("valid", expected_valid)):
            assert dump[key].dtype == np.bool_
            np.testing.assert_array_equal(dump[key], expected)
        expected_gt = np.zeros((2, 8, 8, 3), dtype=np.float16)
        expected_gt[:, 1:-1, 1:-1, 2] = 1
        expected_pred = np.zeros_like(expected_gt)
        expected_pred[0, :, :, 2] = -1
        expected_pred[1, :, :, 0] = 1
        expected_pred[0, 2, 2] = 0
        for key, expected in (("n_gt", expected_gt), ("n_pred", expected_pred)):
            assert dump[key].dtype == np.float16
            np.testing.assert_array_equal(dump[key], expected)
        for key in ("oriented", "unoriented"):
            assert dump[key].shape == (2, 8, 8)
            assert dump[key].dtype == np.float32
            assert np.isnan(dump[key][~expected_valid]).all()
            np.testing.assert_allclose(dump[key][0][expected_valid[0]], 0, atol=1e-3)
            np.testing.assert_allclose(dump[key][1][expected_valid[1]], 90, atol=1e-3)
    manifest = json.loads((tmp_path / "normal_npz" / "manifest.json").read_text())
    assert manifest == {
        "checkpoint": "model.ckpt", "index_path": cfg["dataset"]["view_sampler"]["index_path"],
        "index_md5": hashlib.md5((tmp_path / "index.json").read_bytes()).hexdigest(),
        "num_scenes": 1, "scenes": ["scene00"],
        "experiment": "zeroshot_scannet_pavesplat_vitl",
    }


def test_dump_is_default_off(fake_evaluation, tmp_path):
    stub, batch, _ = fake_evaluation
    assert EvaluationCfg.__dataclass_fields__["save_normal_npz"].default is False
    stub.test_cfg = replace(stub.test_cfg, save_normal_npz=False)
    ModelWrapper.test_step(stub, batch, 0)
    assert not (tmp_path / "normal_npz").exists()
    assert stub._normal_npz_scenes == []


def test_existing_dump_is_preserved(fake_evaluation, tmp_path):
    stub, batch, _ = fake_evaluation
    ModelWrapper.test_step(stub, batch, 0)
    dump = tmp_path / "normal_npz" / "scene00.npz"
    original = dump.read_bytes()
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        ModelWrapper.test_step(stub, batch, 1)
    assert dump.read_bytes() == original
    assert stub._normal_npz_scenes == ["scene00"]


@pytest.mark.parametrize("field,value", [
    ("compute_scores", False), ("eval_normal_error", False), ("render_chunk_size", 1),
])
def test_dump_rejects_incompatible_settings(fake_evaluation, field, value):
    stub, batch, _ = fake_evaluation
    setattr(stub.test_cfg, field, value)
    with pytest.raises(RuntimeError, match=f"test.{field}"):
        ModelWrapper.test_step(stub, batch, 0)
    assert stub._normal_npz_scenes == []
