"""CPU-only tests for physical-scene matched-normal significance analysis."""

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import wilcoxon

from tools.scene_level_wilcoxon import (
    _wilcoxon_cell,
    aggregate_scene_differences,
    analyse_matched_json,
    group_records,
    per_key_test,
)




def _record(scene: str, difference: float) -> dict:
    return {
        "scene": scene,
        "matched_count": 10,
        "median_a_deg": difference,
        "median_b_deg": 0.0,
        "mean_a_deg": difference,
        "mean_b_deg": 0.0,
        "oriented_median_a_deg": difference,
        "oriented_median_b_deg": 0.0,
        "oriented_mean_a_deg": difference,
        "oriented_mean_b_deg": 0.0,
    }


def test_grouping_uses_complete_scene_prefix_and_lists_bad_keys():
    records = [
        _record("scene0707_00_slice00", 1.0),
        _record("scene0707_00_slice12", 2.0),
        _record("scene0707_01_slice00", 3.0),
    ]

    grouped = group_records(records)

    assert list(grouped) == ["scene0707_00", "scene0707_01"]
    assert len(grouped["scene0707_00"]) == 2
    bad_records = records + [
        _record("scene0708_00", 4.0),
        _record("scene0709_00_slice_bad", 5.0),
    ]
    with pytest.raises(ValueError) as exc_info:
        group_records(bad_records)
    assert "scene0708_00" in str(exc_info.value)
    assert "scene0709_00_slice_bad" in str(exc_info.value)


def test_median_and_mean_scene_aggregation_are_hand_computable():
    records = [
        _record("scene_a_slice00", 1.0),
        _record("scene_a_slice01", 3.0),
        _record("scene_a_slice02", 100.0),
        _record("scene_b_slice00", -10.0),
        _record("scene_b_slice01", 2.0),
        _record("scene_b_slice02", 4.0),
    ]

    medians = aggregate_scene_differences(records, "median", "median")
    means = aggregate_scene_differences(records, "median", "mean")

    assert medians == {"scene_a": 3.0, "scene_b": 2.0}
    assert means["scene_a"] == pytest.approx(104.0 / 3.0)
    assert means["scene_b"] == pytest.approx(-4.0 / 3.0)


def test_original_self_check_rejects_perturbed_records():
    perturbed_records = [
        _record(f"scene_{index:02d}_slice00", float(index + 1)) for index in range(12)
    ]

    with pytest.raises(ValueError, match="self-check failed"):
        per_key_test(perturbed_records, expected_pvalue=0.123)


def test_asymptotic_wilcoxon_cell_matches_direct_scipy_call():
    differences = np.arange(1.0, 61.0)
    differences[::2] *= -1.0

    cell = _wilcoxon_cell(differences)
    expected = wilcoxon(
        differences,
        alternative="two-sided",
        method="asymptotic",
        correction=False,
    )

    assert cell["statistic"] == pytest.approx(float(expected.statistic))
    assert cell["pvalue"] == pytest.approx(float(expected.pvalue))
    assert cell["n_scenes"] == 60
    assert cell["win_count_b"] == 30


def test_synthetic_matched_json_aggregates_slices(tmp_path):
    import json
    path = tmp_path / "matched.json"
    path.write_text(json.dumps({"per_scene": {"records": [
        _record("scene_a_slice00", 1.0), _record("scene_a_slice01", 3.0),
        _record("scene_b_slice00", -4.0), _record("scene_c_slice00", 6.0),
    ]}}))
    report = analyse_matched_json(path)
    assert report["n_records"] == 4
    assert report["n_scenes"] == 3
    assert report["primary_per_scene_differences_deg"] == {"scene_a": 2.0, "scene_b": -4.0, "scene_c": 6.0}
    assert report["primary"]["win_count_b"] == 2
