"""CPU-only tests for the matched-pixel rendered-normal analysis tool."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import wilcoxon

from tools.matched_normal_eval import (
    ConsistencyError,
    analyse_runs,
    assert_complete_dump_keys,
    assert_equal_key_sets,
    error_stats,
    file_md5,
    masked_values,
    paired_wilcoxon,
    partition_masks,
    render_markdown,
    validate_scene_pair,
)


def test_mask_partition_bookkeeping_and_count_identities():
    valid_a = np.array([[[True, True, False, False, True]]])
    valid_b = np.array([[[True, False, True, False, True]]])

    masks = partition_masks(valid_a, valid_b)

    assert masks["intersection"].sum() == 2
    assert masks["only_a"].sum() == 1
    assert masks["only_b"].sum() == 1
    assert masks["intersection"].sum() + masks["only_a"].sum() == valid_a.sum()
    assert masks["intersection"].sum() + masks["only_b"].sum() == valid_b.sum()


def test_error_stats_match_direct_numpy_computation():
    unoriented = np.array([5.0, 11.25, 20.0, 29.0, 31.0], dtype=np.float32)
    oriented = np.array([6.0, 12.0, 21.0, 35.0, 40.0], dtype=np.float32)

    result = error_stats(unoriented, oriented)

    assert result["count"] == unoriented.size
    assert result["unoriented_median_deg"] == float(np.median(unoriented))
    assert result["unoriented_mean_deg"] == float(np.mean(unoriented))
    assert result["oriented_median_deg"] == float(np.median(oriented))
    assert result["oriented_mean_deg"] == float(np.mean(oriented))
    assert result["a1125"] == float(np.mean(unoriented < 11.25))
    assert result["a2250"] == float(np.mean(unoriented < 22.5))
    assert result["a30"] == float(np.mean(unoriented < 30.0))


def test_masked_values_use_mask_not_blind_nan_reduction():
    error_map = np.array([[[999.0, 10.0, np.nan, 30.0]]], dtype=np.float32)
    valid = np.array([[[False, True, False, True]]])

    selected = masked_values(error_map, valid, "fixture")

    np.testing.assert_array_equal(selected, np.array([10.0, 30.0], dtype=np.float32))
    assert error_stats(selected, selected)["unoriented_median_deg"] == 20.0


def _pair_fixture():
    shape = (1, 2, 2)
    base = {
        "target_index": np.array([7], dtype=np.int32),
        "valid_gt_range": np.ones(shape, dtype=bool),
        "n_gt": np.zeros(shape + (3,), dtype=np.float16),
    }
    return base, {key: value.copy() for key, value in base.items()}


def test_gt_mismatch_raises():
    dump_a, dump_b = _pair_fixture()
    dump_b["n_gt"][0, 0, 0, 0] = np.float16(0.1)

    with pytest.raises(ConsistencyError, match="n_gt mismatch"):
        validate_scene_pair("scene00", dump_a, dump_b)


def test_index_key_mismatch_raises():
    with pytest.raises(ConsistencyError, match="key sets differ"):
        assert_equal_key_sets({"scene_a", "scene_b"}, {"scene_a", "scene_c"})


@pytest.mark.parametrize(
    "actual, expected",
    [
        ({"scene_a"}, {"scene_a", "scene_b"}),
        ({"scene_a", "stale"}, {"scene_a"}),
    ],
)
def test_completeness_gate_rejects_missing_or_stale_keys(actual, expected):
    with pytest.raises(ConsistencyError, match="incomplete or stale"):
        assert_complete_dump_keys(actual, expected, "fixture")


def test_wilcoxon_is_position_paired_and_records_zero_drops():
    values_a = np.array([1.0, 2.0, 30.0, 4.0, 50.0, 6.0, 70.0, 8.0, 9.0])
    values_b = np.array([2.0, 20.0, 3.0, 40.0, 5.0, 60.0, 7.0, 80.0, 9.0])

    result = paired_wilcoxon(values_a, values_b)
    expected = wilcoxon(values_a, values_b, alternative="two-sided")

    assert result["statistic"] == pytest.approx(float(expected.statistic))
    assert result["pvalue"] == pytest.approx(float(expected.pvalue))
    assert 0.0 <= result["pvalue"] <= 1.0
    assert result["n_pairs"] == len(values_a)
    assert result["n_dropped"] == 1


def test_wilcoxon_all_ties_returns_insignificant_result():
    values = np.array([1.0, 2.5, 7.0, 11.25], dtype=np.float64)

    result = paired_wilcoxon(values, values.copy())

    assert result == {
        "statistic": 0.0,
        "pvalue": 1.0,
        "n_pairs": 4,
        "win_percent_b": 0.0,
        "n_dropped": 4,
        "all_ties": True,
    }


def _write_synthetic_run(
    run_dir: Path,
    index_path: Path,
    checkpoint: str,
    scene_keys: list[str],
    method: str,
    empty_scenes: set[str] | None = None,
) -> None:
    dump_dir = run_dir / "normal_npz"
    metrics_dir = run_dir / "metrics"
    dump_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    all_unoriented = []
    empty_scenes = set() if empty_scenes is None else empty_scenes

    for scene_number, scene in enumerate(scene_keys):
        valid_gt_range = np.array([[[True, True, True, False]]])
        if method == "a":
            valid = np.array([[[True, True, False, False]]])
            finite_errors = np.array([20.0 + scene_number, 40.0 + scene_number])
        else:
            valid = np.array([[[True, False, True, False]]])
            finite_errors = np.array([10.0 + scene_number, 30.0 + scene_number])
        if scene in empty_scenes:
            valid = np.zeros_like(valid)
            finite_errors = np.empty((0,), dtype=np.float32)
        unoriented = np.full(valid.shape, np.nan, dtype=np.float32)
        oriented = np.full(valid.shape, np.nan, dtype=np.float32)
        unoriented[valid] = finite_errors
        oriented[valid] = finite_errors + 1.0
        all_unoriented.append(unoriented[valid])
        normals = np.zeros(valid.shape + (3,), dtype=np.float16)
        normals[..., 2] = np.float16(1.0)
        np.savez_compressed(
            dump_dir / f"{scene}.npz",
            target_index=np.array([100 + scene_number], dtype=np.int32),
            n_pred=normals,
            n_gt=normals,
            unoriented=unoriented,
            oriented=oriented,
            valid=valid,
            valid_gt_range=valid_gt_range,
        )

    manifest = {
        "checkpoint": checkpoint,
        "index_path": str(index_path),
        "index_md5": file_md5(index_path),
        "num_scenes": len(scene_keys),
        "scenes": scene_keys,
        "experiment": f"synthetic_{method}",
    }
    (dump_dir / "manifest.json").write_text(json.dumps(manifest))
    pooled = np.concatenate(all_unoriented)
    scores = {
        "normal_num_pixels": int(pooled.size),
        "normal_unoriented_median_deg": float(np.median(pooled)),
        "normal_unoriented_mean_deg": float(np.mean(pooled)),
    }
    (metrics_dir / "scores_all_avg.json").write_text(json.dumps(scores))


def test_synthetic_runs_join_and_reconcile_end_to_end(tmp_path):
    index_path = tmp_path / "index.json"
    index = {"scene_0": {"target": [1]}, "unused": None, "scene_1": {"target": [2]}}
    index_path.write_text(json.dumps(index))
    scene_keys = ["scene_0", "scene_1"]
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_synthetic_run(run_a, index_path, "baseline.ckpt", scene_keys, "a")
    _write_synthetic_run(run_b, index_path, "core.ckpt", scene_keys, "b")

    report = analyse_runs(run_a, run_b, index_path, "SurfSplat", "PaveSplat")

    assert report["coverage_counts"]["a"] == 4
    assert report["coverage_counts"]["b"] == 4
    assert report["coverage_counts"]["intersection"] == 2
    assert report["symmetric_difference"]["only_a"]["count"] == 2
    assert report["symmetric_difference"]["only_b"]["count"] == 2
    assert report["reconciliation"]["a"]["status"] == "pass"
    assert report["reconciliation"]["b"]["status"] == "pass"
    assert report["per_scene"]["wilcoxon_medians"]["n_pairs"] == 2
    assert "Matched-pixel rendered-normal evaluation" in render_markdown(report)
    json.dumps(report, allow_nan=False)


def test_empty_matched_scene_is_excluded_only_from_per_scene_tests(tmp_path):
    index_path = tmp_path / "index.json"
    scene_keys = ["scene_0", "scene_empty", "scene_2"]
    index_path.write_text(json.dumps({scene: {"target": [1]} for scene in scene_keys}))
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    empty_scenes = {"scene_empty"}
    _write_synthetic_run(
        run_a, index_path, "baseline.ckpt", scene_keys, "a", empty_scenes
    )
    _write_synthetic_run(
        run_b, index_path, "core.ckpt", scene_keys, "b", empty_scenes
    )

    report = analyse_runs(run_a, run_b, index_path, "SurfSplat", "PaveSplat")

    assert report["coverage_counts"]["a"] == 4
    assert report["coverage_counts"]["b"] == 4
    assert report["coverage_counts"]["intersection"] == 2
    assert report["symmetric_difference"]["only_a"]["count"] == 2
    assert report["symmetric_difference"]["only_b"]["count"] == 2
    assert report["per_scene"]["wilcoxon_medians"]["n_pairs"] == 2
    assert report["per_scene"]["empty_matched_scenes"] == [
        {
            "scene": "scene_empty",
            "own_a_count": 0,
            "own_b_count": 0,
            "gt_range_count": 3,
        }
    ]
    markdown = render_markdown(report)
    assert "`scene_empty`" in markdown
    assert "excluded from the per-scene Wilcoxon tests" in markdown
