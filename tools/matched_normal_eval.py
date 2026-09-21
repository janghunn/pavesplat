#!/usr/bin/env python3
"""Rebuttal Table A: compare rendered normals on exactly matched valid pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import wilcoxon


EXPECTED_FIELDS = {
    "target_index",
    "n_pred",
    "n_gt",
    "unoriented",
    "oriented",
    "valid",
    "valid_gt_range",
}
RECONCILIATION_ATOL = 5e-3


class ConsistencyError(RuntimeError):
    """Raised when two runs cannot support an exact matched-pixel comparison."""


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index_keys(path: Path) -> list[str]:
    try:
        with path.open() as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConsistencyError(f"Could not read evaluation index {path}: {exc}") from exc
    if not isinstance(index, dict):
        raise ConsistencyError(f"Evaluation index must be a JSON object: {path}")
    keys = [key for key, value in index.items() if value is not None]
    if not keys:
        raise ConsistencyError(f"Evaluation index has no non-null entries: {path}")
    return keys


def discover_dump_keys(run_dir: Path) -> set[str]:
    dump_dir = run_dir / "normal_npz"
    if not dump_dir.is_dir():
        raise ConsistencyError(f"Missing normal_npz directory: {dump_dir}")
    return {path.stem for path in dump_dir.glob("*.npz")}


def _short_keys(keys: Iterable[str], limit: int = 8) -> str:
    values = sorted(keys)
    suffix = "" if len(values) <= limit else f" ... (+{len(values) - limit})"
    return ", ".join(values[:limit]) + suffix


def assert_complete_dump_keys(
    actual: set[str], expected: set[str], label: str
) -> None:
    if actual == expected:
        return
    missing = expected - actual
    unexpected = actual - expected
    raise ConsistencyError(
        f"{label} dump is incomplete or stale: expected {len(expected)} keys, "
        f"found {len(actual)}; missing=[{_short_keys(missing)}]; "
        f"unexpected=[{_short_keys(unexpected)}]"
    )


def assert_equal_key_sets(keys_a: set[str], keys_b: set[str]) -> None:
    if keys_a != keys_b:
        raise ConsistencyError(
            "Run dump key sets differ: "
            f"only A=[{_short_keys(keys_a - keys_b)}]; "
            f"only B=[{_short_keys(keys_b - keys_a)}]"
        )


def load_manifest(run_dir: Path, dump_keys: set[str]) -> dict:
    path = run_dir / "normal_npz" / "manifest.json"
    try:
        with path.open() as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConsistencyError(f"Could not read manifest {path}: {exc}") from exc

    required = {"checkpoint", "index_path", "index_md5", "num_scenes", "scenes"}
    missing = required - set(manifest)
    if missing:
        raise ConsistencyError(f"Manifest {path} is missing: {_short_keys(missing)}")
    scenes = manifest["scenes"]
    if not isinstance(scenes, list) or not all(isinstance(x, str) for x in scenes):
        raise ConsistencyError(f"Manifest scenes must be a string list: {path}")
    if len(scenes) != len(set(scenes)):
        raise ConsistencyError(f"Manifest contains duplicate scene keys: {path}")
    if manifest["num_scenes"] != len(scenes):
        raise ConsistencyError(
            f"Manifest num_scenes does not match its scene list: {path}"
        )
    if set(scenes) != dump_keys:
        raise ConsistencyError(
            f"Manifest scene list does not match .npz files in {run_dir}"
        )
    return manifest


def validate_manifests(
    manifest_a: dict, manifest_b: dict, index_path: Path
) -> None:
    if manifest_a["index_path"] != manifest_b["index_path"]:
        raise ConsistencyError(
            "Manifest index_path values differ: "
            f"{manifest_a['index_path']!r} vs {manifest_b['index_path']!r}"
        )
    if manifest_a["index_md5"] != manifest_b["index_md5"]:
        raise ConsistencyError(
            "Manifest evaluation-index MD5 values differ: "
            f"{manifest_a['index_md5']} vs {manifest_b['index_md5']}"
        )
    actual_md5 = file_md5(index_path)
    if manifest_a["index_md5"] != actual_md5:
        raise ConsistencyError(
            "Manifest evaluation-index MD5 does not match --index: "
            f"{manifest_a['index_md5']} vs {actual_md5}"
        )
    checkpoint_a = manifest_a["checkpoint"]
    checkpoint_b = manifest_b["checkpoint"]
    if not checkpoint_a or not checkpoint_b:
        raise ConsistencyError("Both manifests must identify checkpointing.load")
    if checkpoint_a == checkpoint_b:
        raise ConsistencyError(
            f"The two manifests identify the same checkpoint: {checkpoint_a}"
        )


def load_dump(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = set(archive.files)
            if fields != EXPECTED_FIELDS:
                raise ConsistencyError(
                    f"{path} fields differ from the dump contract; "
                    f"missing=[{_short_keys(EXPECTED_FIELDS - fields)}]; "
                    f"unexpected=[{_short_keys(fields - EXPECTED_FIELDS)}]"
                )
            dump = {name: archive[name] for name in EXPECTED_FIELDS}
    except (OSError, ValueError) as exc:
        raise ConsistencyError(f"Could not load {path}: {exc}") from exc

    target_index = dump["target_index"]
    error_shape = dump["unoriented"].shape
    if target_index.dtype != np.int32 or target_index.ndim != 1:
        raise ConsistencyError(f"{path}: target_index must be int32 [V]")
    if dump["unoriented"].dtype != np.float32 or len(error_shape) != 3:
        raise ConsistencyError(f"{path}: unoriented must be float32 [V,H,W]")
    if dump["oriented"].dtype != np.float32 or dump["oriented"].shape != error_shape:
        raise ConsistencyError(f"{path}: oriented must be float32 [V,H,W]")
    if error_shape[0] != target_index.shape[0]:
        raise ConsistencyError(f"{path}: target_index length does not match V")
    normal_shape = error_shape + (3,)
    for key in ("n_pred", "n_gt"):
        if dump[key].dtype != np.float16 or dump[key].shape != normal_shape:
            raise ConsistencyError(f"{path}: {key} must be float16 [V,H,W,3]")
        if not np.isfinite(dump[key]).all():
            raise ConsistencyError(f"{path}: {key} contains non-finite values")
    for key in ("valid", "valid_gt_range"):
        if dump[key].dtype != np.bool_ or dump[key].shape != error_shape:
            raise ConsistencyError(f"{path}: {key} must be bool [V,H,W]")
    if np.any(dump["valid"] & ~dump["valid_gt_range"]):
        raise ConsistencyError(f"{path}: valid is not a subset of valid_gt_range")
    for key in ("unoriented", "oriented"):
        if not np.isfinite(dump[key][dump["valid"]]).all():
            raise ConsistencyError(f"{path}: {key} is non-finite on valid pixels")
        if not np.isnan(dump[key][~dump["valid"]]).all():
            raise ConsistencyError(f"{path}: {key} must be NaN outside valid")
    return dump


def validate_scene_pair(scene: str, dump_a: dict, dump_b: dict) -> None:
    if not np.array_equal(dump_a["target_index"], dump_b["target_index"]):
        raise ConsistencyError(f"{scene}: target_index mismatch")
    if not np.array_equal(dump_a["valid_gt_range"], dump_b["valid_gt_range"]):
        raise ConsistencyError(f"{scene}: valid_gt_range mismatch")
    if not np.allclose(dump_a["n_gt"], dump_b["n_gt"], rtol=1e-3, atol=1e-3):
        raise ConsistencyError(f"{scene}: n_gt mismatch beyond float16 tolerance")


def partition_masks(valid_a: np.ndarray, valid_b: np.ndarray) -> dict[str, np.ndarray]:
    if valid_a.dtype != np.bool_ or valid_b.dtype != np.bool_:
        raise ConsistencyError("Coverage masks must be boolean")
    if valid_a.shape != valid_b.shape:
        raise ConsistencyError(
            f"Coverage-mask shapes differ: {valid_a.shape} vs {valid_b.shape}"
        )
    return {
        "own_a": valid_a,
        "own_b": valid_b,
        "intersection": valid_a & valid_b,
        "only_a": valid_a & ~valid_b,
        "only_b": valid_b & ~valid_a,
    }


def masked_values(error_map: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    if error_map.shape != mask.shape:
        raise ConsistencyError(
            f"{label}: error-map and mask shapes differ: {error_map.shape} vs {mask.shape}"
        )
    values = error_map[mask]
    if not np.isfinite(values).all():
        raise ConsistencyError(f"{label}: selected error values contain NaN or infinity")
    return values


def error_stats(unoriented: np.ndarray, oriented: np.ndarray) -> dict:
    if unoriented.ndim != 1 or oriented.ndim != 1:
        raise ConsistencyError("Error-stat inputs must be one-dimensional")
    if unoriented.shape != oriented.shape:
        raise ConsistencyError("Oriented and unoriented error counts differ")
    count = int(unoriented.size)
    if count == 0:
        return {
            "count": 0,
            "unoriented_median_deg": None,
            "unoriented_mean_deg": None,
            "a1125": None,
            "a2250": None,
            "a30": None,
            "oriented_median_deg": None,
            "oriented_mean_deg": None,
        }
    if not np.isfinite(unoriented).all() or not np.isfinite(oriented).all():
        raise ConsistencyError("Error-stat inputs must be finite")
    return {
        "count": count,
        "unoriented_median_deg": float(np.median(unoriented)),
        "unoriented_mean_deg": float(np.mean(unoriented)),
        "a1125": float(np.mean(unoriented < 11.25)),
        "a2250": float(np.mean(unoriented < 22.5)),
        "a30": float(np.mean(unoriented < 30.0)),
        "oriented_median_deg": float(np.median(oriented)),
        "oriented_mean_deg": float(np.mean(oriented)),
    }


def paired_difference_stats(
    unoriented_a: np.ndarray,
    unoriented_b: np.ndarray,
    oriented_a: np.ndarray,
    oriented_b: np.ndarray,
) -> dict:
    if not (
        unoriented_a.shape
        == unoriented_b.shape
        == oriented_a.shape
        == oriented_b.shape
    ):
        raise ConsistencyError("Paired matched-error arrays have different shapes")
    if unoriented_a.size == 0:
        raise ConsistencyError("The global matched-pixel set is empty")
    delta_un = unoriented_a - unoriented_b
    delta_or = oriented_a - oriented_b
    return {
        "definition": "A minus B at each matched pixel (positive favors B)",
        "count": int(delta_un.size),
        "unoriented_mean_deg": float(np.mean(delta_un)),
        "unoriented_median_deg": float(np.median(delta_un)),
        "oriented_mean_deg": float(np.mean(delta_or)),
        "oriented_median_deg": float(np.median(delta_or)),
    }


def paired_wilcoxon(values_a: Sequence[float], values_b: Sequence[float]) -> dict:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ConsistencyError("Wilcoxon inputs must be non-empty, position-paired vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ConsistencyError("Wilcoxon inputs must be finite")
    differences = a - b
    win_percent_b = float(100.0 * np.mean(b < a))
    if np.all(differences == 0):
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "n_pairs": int(a.size),
            "win_percent_b": win_percent_b,
            "n_dropped": int(a.size),
            "all_ties": True,
        }
    result = wilcoxon(a, b, alternative="two-sided")
    summary = {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "n_pairs": int(a.size),
        "win_percent_b": win_percent_b,
    }
    n_dropped = int(np.count_nonzero(differences == 0))
    if n_dropped:
        summary["n_dropped"] = n_dropped
    return summary


def _bucket() -> dict[str, list[np.ndarray]]:
    return {"unoriented": [], "oriented": []}


def _append_bucket(
    bucket: dict[str, list[np.ndarray]], dump: dict, mask: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray]:
    unoriented = masked_values(dump["unoriented"], mask, f"{label} unoriented")
    oriented = masked_values(dump["oriented"], mask, f"{label} oriented")
    bucket["unoriented"].append(unoriented)
    bucket["oriented"].append(oriented)
    return unoriented, oriented


def _concatenate(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(chunks)


def reconcile_metrics(
    run_dir: Path,
    name: str,
    own_stats: dict,
    allow_mismatch: bool,
) -> dict:
    score_path = run_dir / "metrics" / "scores_all_avg.json"
    if not score_path.is_file():
        return {"status": "not_available", "path": str(score_path)}
    try:
        with score_path.open() as f:
            scores = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConsistencyError(f"Could not read {score_path}: {exc}") from exc

    expected = {
        "normal_num_pixels": own_stats["count"],
        "normal_unoriented_median_deg": own_stats["unoriented_median_deg"],
        "normal_unoriented_mean_deg": own_stats["unoriented_mean_deg"],
    }
    missing = set(expected) - set(scores)
    discrepancies = []
    if missing:
        discrepancies.append(f"missing score keys: {_short_keys(missing)}")
    if not missing:
        if int(scores["normal_num_pixels"]) != expected["normal_num_pixels"]:
            discrepancies.append(
                "normal_num_pixels "
                f"dump={expected['normal_num_pixels']} scores={scores['normal_num_pixels']}"
            )
        for key in (
            "normal_unoriented_median_deg",
            "normal_unoriented_mean_deg",
        ):
            difference = abs(float(scores[key]) - float(expected[key]))
            if difference > RECONCILIATION_ATOL:
                discrepancies.append(
                    f"{key} dump={expected[key]:.9g} scores={float(scores[key]):.9g} "
                    f"abs_diff={difference:.9g}"
                )

    status = "pass" if not discrepancies else "allowed_mismatch"
    if discrepancies:
        message = f"RECONCILIATION MISMATCH for {name}: " + "; ".join(discrepancies)
        print(message, file=sys.stderr)
        if not allow_mismatch:
            raise ConsistencyError(message)
    return {
        "status": status,
        "path": str(score_path),
        "tolerance": RECONCILIATION_ATOL,
        "discrepancies": discrepancies,
    }


def classify_verdict(own_a: dict, own_b: dict, inter_a: dict, inter_b: dict, pvalue: float) -> dict:
    delta_own = own_a["unoriented_median_deg"] - own_b["unoriented_median_deg"]
    delta_matched = inter_a["unoriented_median_deg"] - inter_b["unoriented_median_deg"]
    significant_b_better = pvalue < 0.05 and delta_matched > 0
    if not significant_b_better:
        letter = "C"
        reading = "matched-set medians are not significantly better for B"
    elif delta_matched >= 0.8 * delta_own:
        letter = "A"
        reading = "coverage is not the driver"
    else:
        letter = "B"
        reading = "coverage partly explains the own-coverage reduction"

    retention = None if delta_own == 0 else delta_matched / delta_own
    boundary_note = None
    if retention is not None and abs(retention - 0.8) <= 0.05:
        boundary_note = (
            "Retention is within 0.05 of the predeclared 0.8 A/B boundary; "
            "report both adjacent readings."
        )
    return {
        "letter": letter,
        "reading": reading,
        "delta_own_median_deg_a_minus_b": delta_own,
        "delta_matched_median_deg_a_minus_b": delta_matched,
        "matched_to_own_reduction_ratio": retention,
        "wilcoxon_alpha": 0.05,
        "retention_threshold": 0.8,
        "boundary_note": boundary_note,
    }


def analyse_runs(
    run_a: Path,
    run_b: Path,
    index_path: Path,
    name_a: str,
    name_b: str,
    allow_mismatch: bool = False,
) -> dict:
    index_keys = load_index_keys(index_path)
    expected_keys = set(index_keys)
    keys_a = discover_dump_keys(run_a)
    keys_b = discover_dump_keys(run_b)
    assert_equal_key_sets(keys_a, keys_b)
    assert_complete_dump_keys(keys_a, expected_keys, name_a)
    assert_complete_dump_keys(keys_b, expected_keys, name_b)

    manifest_a = load_manifest(run_a, keys_a)
    manifest_b = load_manifest(run_b, keys_b)
    validate_manifests(manifest_a, manifest_b, index_path)

    buckets = {
        "own_a": _bucket(),
        "own_b": _bucket(),
        "intersection_a": _bucket(),
        "intersection_b": _bucket(),
        "only_a": _bucket(),
        "only_b": _bucket(),
    }
    per_scene = []
    empty_matched_scenes = []

    for scene in index_keys:
        dump_a = load_dump(run_a / "normal_npz" / f"{scene}.npz")
        dump_b = load_dump(run_b / "normal_npz" / f"{scene}.npz")
        validate_scene_pair(scene, dump_a, dump_b)
        masks = partition_masks(dump_a["valid"], dump_b["valid"])

        _append_bucket(buckets["own_a"], dump_a, masks["own_a"], f"{scene} own A")
        _append_bucket(buckets["own_b"], dump_b, masks["own_b"], f"{scene} own B")
        inter_a_un, inter_a_or = _append_bucket(
            buckets["intersection_a"], dump_a, masks["intersection"], f"{scene} matched A"
        )
        inter_b_un, inter_b_or = _append_bucket(
            buckets["intersection_b"], dump_b, masks["intersection"], f"{scene} matched B"
        )
        _append_bucket(buckets["only_a"], dump_a, masks["only_a"], f"{scene} only A")
        _append_bucket(buckets["only_b"], dump_b, masks["only_b"], f"{scene} only B")

        if inter_a_un.size == 0:
            empty_matched_scenes.append(
                {
                    "scene": scene,
                    "own_a_count": int(np.count_nonzero(masks["own_a"])),
                    "own_b_count": int(np.count_nonzero(masks["own_b"])),
                    "gt_range_count": int(
                        np.count_nonzero(dump_a["valid_gt_range"])
                    ),
                }
            )
            continue
        per_scene.append(
            {
                "scene": scene,
                "matched_count": int(inter_a_un.size),
                "median_a_deg": float(np.median(inter_a_un)),
                "median_b_deg": float(np.median(inter_b_un)),
                "mean_a_deg": float(np.mean(inter_a_un)),
                "mean_b_deg": float(np.mean(inter_b_un)),
                "oriented_median_a_deg": float(np.median(inter_a_or)),
                "oriented_median_b_deg": float(np.median(inter_b_or)),
                "oriented_mean_a_deg": float(np.mean(inter_a_or)),
                "oriented_mean_b_deg": float(np.mean(inter_b_or)),
            }
        )

    if not per_scene:
        raise ConsistencyError("Every scene has an empty matched-pixel set")

    arrays = {}
    for label, bucket in buckets.items():
        arrays[label] = {
            "unoriented": _concatenate(bucket["unoriented"]),
            "oriented": _concatenate(bucket["oriented"]),
        }
        bucket["unoriented"].clear()
        bucket["oriented"].clear()

    stats = {
        label: error_stats(values["unoriented"], values["oriented"])
        for label, values in arrays.items()
    }
    paired_difference = paired_difference_stats(
        arrays["intersection_a"]["unoriented"],
        arrays["intersection_b"]["unoriented"],
        arrays["intersection_a"]["oriented"],
        arrays["intersection_b"]["oriented"],
    )

    median_wilcoxon = paired_wilcoxon(
        [row["median_a_deg"] for row in per_scene],
        [row["median_b_deg"] for row in per_scene],
    )
    mean_wilcoxon = paired_wilcoxon(
        [row["mean_a_deg"] for row in per_scene],
        [row["mean_b_deg"] for row in per_scene],
    )
    oriented_median_wilcoxon = paired_wilcoxon(
        [row["oriented_median_a_deg"] for row in per_scene],
        [row["oriented_median_b_deg"] for row in per_scene],
    )
    oriented_mean_wilcoxon = paired_wilcoxon(
        [row["oriented_mean_a_deg"] for row in per_scene],
        [row["oriented_mean_b_deg"] for row in per_scene],
    )

    count_a = stats["own_a"]["count"]
    count_b = stats["own_b"]["count"]
    count_inter = stats["intersection_a"]["count"]
    if count_inter + stats["only_a"]["count"] != count_a:
        raise ConsistencyError("Count identity failed: |A∩B| + |A\\B| != |A|")
    if count_inter + stats["only_b"]["count"] != count_b:
        raise ConsistencyError("Count identity failed: |A∩B| + |B\\A| != |B|")

    reconciliation = {
        "a": reconcile_metrics(run_a, name_a, stats["own_a"], allow_mismatch),
        "b": reconcile_metrics(run_b, name_b, stats["own_b"], allow_mismatch),
    }
    verdict = classify_verdict(
        stats["own_a"],
        stats["own_b"],
        stats["intersection_a"],
        stats["intersection_b"],
        median_wilcoxon["pvalue"],
    )

    return {
        "schema_version": 1,
        "methods": {"a": name_a, "b": name_b},
        "runs": {"a": str(run_a), "b": str(run_b)},
        "index": {
            "path": str(index_path),
            "md5": file_md5(index_path),
            "non_null_keys": len(index_keys),
        },
        "manifests": {"a": manifest_a, "b": manifest_b},
        "own_coverage": {"a": stats["own_a"], "b": stats["own_b"]},
        "intersection": {
            "a": stats["intersection_a"],
            "b": stats["intersection_b"],
            "paired_difference": paired_difference,
        },
        "symmetric_difference": {
            "only_a": stats["only_a"],
            "only_b": stats["only_b"],
        },
        "coverage_counts": {
            "a": count_a,
            "b": count_b,
            "intersection": count_inter,
            "intersection_over_a": count_inter / count_a,
            "intersection_over_b": count_inter / count_b,
        },
        "per_scene": {
            "granularity": "one position-paired scalar per non-null evaluation-index key",
            "records": per_scene,
            "empty_matched_scenes": empty_matched_scenes,
            "wilcoxon_medians": median_wilcoxon,
            "wilcoxon_means": mean_wilcoxon,
            "wilcoxon_oriented_medians": oriented_median_wilcoxon,
            "wilcoxon_oriented_means": oriented_mean_wilcoxon,
        },
        "reconciliation": reconciliation,
        "verdict": verdict,
    }


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def render_markdown(report: dict) -> str:
    name_a = report["methods"]["a"]
    name_b = report["methods"]["b"]
    own = report["own_coverage"]
    inter = report["intersection"]
    only = report["symmetric_difference"]
    counts = report["coverage_counts"]
    med_test = report["per_scene"]["wilcoxon_medians"]
    mean_test = report["per_scene"]["wilcoxon_means"]
    ori_med_test = report["per_scene"]["wilcoxon_oriented_medians"]
    ori_mean_test = report["per_scene"]["wilcoxon_oriented_means"]
    verdict = report["verdict"]

    rows = []
    for coverage, method, values in (
        ("own", name_a, own["a"]),
        ("own", name_b, own["b"]),
        ("intersection", name_a, inter["a"]),
        ("intersection", name_b, inter["b"]),
        ("A only", name_a, only["only_a"]),
        ("B only", name_b, only["only_b"]),
    ):
        rows.append(
            f"| {coverage} | {method} | {_fmt(values['count'])} | "
            f"{_fmt(values['unoriented_median_deg'])} | "
            f"{_fmt(values['unoriented_mean_deg'])} | {_fmt(values['a1125'])} | "
            f"{_fmt(values['a2250'])} | {_fmt(values['a30'])} | "
            f"{_fmt(values['oriented_median_deg'])} | {_fmt(values['oriented_mean_deg'])} |"
        )

    boundary = ""
    if verdict["boundary_note"]:
        boundary = f"\n\n**Boundary warning:** {verdict['boundary_note']}"
    empty_note = ""
    empty_scenes = report["per_scene"]["empty_matched_scenes"]
    if empty_scenes:
        scene_names = ", ".join(f"`{row['scene']}`" for row in empty_scenes)
        empty_note = (
            "- Per-scene test exclusion: "
            f"{scene_names} have empty matched sets and are excluded from the "
            "per-scene Wilcoxon tests; pooled statistics still include their pixels.\n"
        )
    return f"""# Matched-pixel rendered-normal evaluation

- A: **{name_a}** — `{report['manifests']['a']['checkpoint']}`
- B: **{name_b}** — `{report['manifests']['b']['checkpoint']}`
- Evaluation index: `{report['index']['path']}` ({report['index']['non_null_keys']} non-null keys, MD5 `{report['index']['md5']}`)

| coverage set | method | pixels | unor. median ° | unor. mean ° | a11.25 | a22.5 | a30 | ori. median ° | ori. mean ° |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Coverage counts

- |A| = {_fmt(counts['a'])}; |B| = {_fmt(counts['b'])}; |A∩B| = {_fmt(counts['intersection'])}.
- |A∩B|/|A| = {100 * counts['intersection_over_a']:.6f}%; |A∩B|/|B| = {100 * counts['intersection_over_b']:.6f}%.

## Paired results

{empty_note}- Per-pixel A−B matched difference: median {inter['paired_difference']['unoriented_median_deg']:.6f}°, mean {inter['paired_difference']['unoriented_mean_deg']:.6f}° (positive favors B).
- Per-key median Wilcoxon: statistic={med_test['statistic']:.6g}, p={med_test['pvalue']:.6g}, B win={med_test['win_percent_b']:.3f}%, n={med_test['n_pairs']}.
- Per-key mean Wilcoxon: statistic={mean_test['statistic']:.6g}, p={mean_test['pvalue']:.6g}, B win={mean_test['win_percent_b']:.3f}%, n={mean_test['n_pairs']}.
- Per-key oriented-median Wilcoxon (secondary): statistic={ori_med_test['statistic']:.6g}, p={ori_med_test['pvalue']:.6g}, B win={ori_med_test['win_percent_b']:.3f}%, n={ori_med_test['n_pairs']}.
- Per-key oriented-mean Wilcoxon (secondary): statistic={ori_mean_test['statistic']:.6g}, p={ori_mean_test['pvalue']:.6g}, B win={ori_mean_test['win_percent_b']:.3f}%, n={ori_mean_test['n_pairs']}.

## Verdict

**{verdict['letter']} — {verdict['reading']}.** Δown={verdict['delta_own_median_deg_a_minus_b']:.6f}°, Δmatched={verdict['delta_matched_median_deg_a_minus_b']:.6f}°, retention={_fmt(verdict['matched_to_own_reduction_ratio'], 6)}.{boundary}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="report scores_all_avg.json reconciliation mismatches instead of failing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = analyse_runs(
            args.run_a,
            args.run_b,
            args.index,
            args.name_a,
            args.name_b,
            allow_mismatch=args.allow_mismatch,
        )
        markdown = render_markdown(report)
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as f:
            json.dump(report, f, indent=2, allow_nan=False)
        args.out_md.write_text(markdown)
    except ConsistencyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(markdown)
    print(f"Wrote JSON report to {args.out_json}")
    print(f"Wrote Markdown report to {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
