#!/usr/bin/env python3
"""Rebuttal Table A: matched-normal Wilcoxon tests per physical scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import scipy
from scipy.stats import wilcoxon


SCENE_KEY_RE = re.compile(r"^(?P<scene>.+)_slice\d+$")
SELF_CHECK_RELATIVE_TOLERANCE = 1e-6
ASYMPTOTIC_METHOD_LABEL = "normal approximation (correction=False)"
EXACT_METHOD_LABEL = "exact signed-rank distribution"
FAMILY_FIELDS = {
    "median": ("median_a_deg", "median_b_deg"),
    "mean": ("mean_a_deg", "mean_b_deg"),
    "oriented_median": ("oriented_median_a_deg", "oriented_median_b_deg"),
    "oriented_mean": ("oriented_mean_a_deg", "oriented_mean_b_deg"),
}
AGGREGATIONS = ("median", "mean")


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: Path) -> list[dict]:
    with path.open() as f:
        payload = json.load(f)
    try:
        records = payload["per_scene"]["records"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path} does not contain per_scene.records") from exc
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: per_scene.records must be a non-empty list")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path}: every per_scene.records item must be an object")
    return records


def physical_scene_id(key: str) -> str:
    match = SCENE_KEY_RE.fullmatch(key)
    if match is None:
        raise ValueError(f"Evaluation-index key does not match {SCENE_KEY_RE.pattern!r}: {key!r}")
    return match.group("scene")


def group_records(records: Sequence[Mapping]) -> dict[str, list[Mapping]]:
    offenders = []
    for record in records:
        key = record.get("scene")
        if not isinstance(key, str) or SCENE_KEY_RE.fullmatch(key) is None:
            offenders.append(repr(key))
    if offenders:
        raise ValueError(
            f"Evaluation-index keys not matching {SCENE_KEY_RE.pattern!r}: "
            + ", ".join(sorted(offenders))
        )

    grouped: dict[str, list[Mapping]] = defaultdict(list)
    seen_keys = set()
    for record in records:
        key = record["scene"]
        if key in seen_keys:
            raise ValueError(f"Duplicate per-key record: {key!r}")
        seen_keys.add(key)
        grouped[physical_scene_id(key)].append(record)
    return {scene: grouped[scene] for scene in sorted(grouped)}


def _numeric_field(record: Mapping, field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{record.get('scene')!r}: {field} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{record.get('scene')!r}: {field} must be finite")
    return result


def _aggregate_grouped_differences(
    grouped: Mapping[str, Sequence[Mapping]], family: str, aggregation: str
) -> dict[str, float]:
    if family not in FAMILY_FIELDS:
        raise ValueError(f"Unknown statistic family: {family!r}")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unknown scene aggregation: {aggregation!r}")
    field_a, field_b = FAMILY_FIELDS[family]
    reducer = np.median if aggregation == "median" else np.mean
    result = {}
    for scene in sorted(grouped):
        differences = np.asarray(
            [
                _numeric_field(record, field_a) - _numeric_field(record, field_b)
                for record in grouped[scene]
            ],
            dtype=np.float64,
        )
        result[scene] = float(reducer(differences))
    return result


def aggregate_scene_differences(
    records: Sequence[Mapping], family: str, aggregation: str
) -> dict[str, float]:
    """Aggregate slice-level A-minus-B differences once per physical scene."""
    return _aggregate_grouped_differences(
        group_records(records), family=family, aggregation=aggregation
    )


def _wilcoxon_cell(differences: Sequence[float]) -> dict:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Wilcoxon differences must be a non-empty finite vector")
    zero_count = int(np.count_nonzero(values == 0))
    if zero_count == values.size:
        statistic, pvalue = 0.0, 1.0
    else:
        result = wilcoxon(
            values,
            alternative="two-sided",
            method="asymptotic",
            correction=False,
        )
        statistic, pvalue = float(result.statistic), float(result.pvalue)
    win_count = int(np.count_nonzero(values > 0))
    loss_count = int(np.count_nonzero(values < 0))
    return {
        "method": "asymptotic",
        "method_label": ASYMPTOTIC_METHOD_LABEL,
        "alternative": "two-sided",
        "statistic": statistic,
        "pvalue": pvalue,
        "n_scenes": int(values.size),
        "win_count_b": win_count,
        "win_percent_b": float(100.0 * win_count / values.size),
        "loss_count_b": loss_count,
        "zero_difference_count": zero_count,
        "median_difference_deg": float(np.median(values)),
        "mean_difference_deg": float(np.mean(values)),
    }


def per_key_test(records: Sequence[Mapping], expected_pvalue: float | None = None) -> dict:
    values_a = np.asarray(
        [_numeric_field(record, "median_a_deg") for record in records],
        dtype=np.float64,
    )
    values_b = np.asarray(
        [_numeric_field(record, "median_b_deg") for record in records],
        dtype=np.float64,
    )
    cell = _wilcoxon_cell(values_a - values_b)
    pvalue = cell['pvalue']
    if expected_pvalue is not None and not np.isclose(
        pvalue, expected_pvalue, rtol=SELF_CHECK_RELATIVE_TOLERANCE, atol=0.0
    ):
        raise ValueError(f"Per-key self-check failed: expected {expected_pvalue}, got {pvalue}")
    return {**cell, "n_records": int(values_a.size), "expected_pvalue": expected_pvalue}


def _absolute_tie_count(values: np.ndarray) -> int:
    nonzero_magnitudes = np.abs(values[values != 0])
    counts = Counter(nonzero_magnitudes.tolist())
    return int(sum(count - 1 for count in counts.values() if count > 1))


def analyse_matched_json(path: Path, expected_pvalue: float | None = None) -> dict:
    records = load_records(path)
    grouped = group_records(records)
    self_check = per_key_test(records, expected_pvalue)

    aggregated = {}
    tests = {}
    for family in FAMILY_FIELDS:
        aggregated[family] = {}
        tests[family] = {}
        for aggregation in AGGREGATIONS:
            scene_values = _aggregate_grouped_differences(grouped, family, aggregation)
            aggregated[family][aggregation] = scene_values
            tests[family][aggregation] = _wilcoxon_cell(list(scene_values.values()))

    primary_scene_values = aggregated["median"]["median"]
    primary_values = np.asarray(list(primary_scene_values.values()), dtype=np.float64)
    zero_count = int(np.count_nonzero(primary_values == 0))
    absolute_tie_count = _absolute_tie_count(primary_values)
    exact_result = wilcoxon(
        primary_values,
        alternative="two-sided",
        method="exact",
        correction=False,
    )
    primary = {
        "family": "median",
        "aggregation": "median",
        **tests["median"]["median"],
        "asymptotic_pvalue": tests["median"]["median"]["pvalue"],
        "exact_method": "exact",
        "exact_method_label": EXACT_METHOD_LABEL,
        "exact_pvalue": float(exact_result.pvalue),
        "absolute_difference_tie_count": absolute_tie_count,
        "exact_clean": zero_count == 0 and absolute_tie_count == 0,
    }

    slice_counts = {scene: len(grouped[scene]) for scene in sorted(grouped)}
    histogram = Counter(slice_counts.values())
    return {
        "schema_version": 1,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "input": {"path": str(path), "md5": file_md5(path)},
        "difference_definition": "slice-level A minus B in degrees (positive favors B/PaveSplat)",
        "primary_aggregation_rationale": (
            "The per-scene median is robust to outlier slices and is coherent with the "
            "primary per-key median-error statistic."
        ),
        "n_records": len(records),
        "n_scenes": len(grouped),
        "slices_per_scene_histogram": {
            str(count): histogram[count] for count in sorted(histogram)
        },
        "slice_counts_by_scene": slice_counts,
        "primary_per_scene_differences_deg": primary_scene_values,
        "primary": primary,
        "tests": tests,
        "per_key_test": self_check,
    }


def render_markdown(report: Mapping) -> str:
    primary = report["primary"]
    self_check = report["per_key_test"]
    lines = [
        "# Physical-scene matched-normal Wilcoxon analysis",
        "",
        "## Headline",
        "",
        (
            "- Primary value: median over each physical scene's slice-level paired "
            "differences in matched-pixel median unoriented normal error (A−B; positive "
            "favors B/PaveSplat)."
        ),
        f"- Aggregation rationale: {report['primary_aggregation_rationale']}",
        (
            f"- **N={primary['n_scenes']} physical scenes; two-sided Wilcoxon "
            f"W={primary['statistic']:.0f}, p={primary['asymptotic_pvalue']:.9g} "
            f"({primary['method_label']}).**"
        ),
        (
            f"- Exact p={primary['exact_pvalue']:.9g} "
            f"({primary['exact_method_label']}); exact-clean={str(primary['exact_clean']).lower()} "
            f"(zero differences={primary['zero_difference_count']}, tied |differences|="
            f"{primary['absolute_difference_tie_count']})."
        ),
        (
            f"- Method B wins: {primary['win_count_b']}/{primary['n_scenes']} "
            f"({primary['win_percent_b']:.1f}%); median scene difference "
            f"{primary['median_difference_deg']:+.6f}°, mean "
            f"{primary['mean_difference_deg']:+.6f}°."
        ),
        (
            f"- Per-key comparison: n={self_check['n_records']}, "
            f"W={self_check['statistic']:.0f}, p={self_check['pvalue']:.17g} "
            f"({self_check['method_label']})."
        ),
        f"- Input: `{report['input']['path']}` (MD5 `{report['input']['md5']}`).",
        (
            f"- Versions: numpy {report['versions']['numpy']}; "
            f"scipy {report['versions']['scipy']}."
        ),
        "",
        "## Primary per-scene values",
        "",
        "| physical scene | slices | median A−B (deg) |",
        "|---|---:|---:|",
    ]
    for scene, difference in report["primary_per_scene_differences_deg"].items():
        lines.append(
            f"| {scene} | {report['slice_counts_by_scene'][scene]} | {difference:+.9f} |"
        )

    lines.extend(
        [
            "",
            "## Variants",
            "",
            "All asymptotic cells use a two-sided normal approximation with `correction=False`.",
            "",
            "| statistic family | scene aggregation | N | W | p | Method B wins | median A−B (deg) | mean A−B (deg) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family in FAMILY_FIELDS:
        for aggregation in AGGREGATIONS:
            cell = report["tests"][family][aggregation]
            lines.append(
                f"| {family} | {aggregation} | {cell['n_scenes']} | "
                f"{cell['statistic']:.6g} | {cell['pvalue']:.9g} | "
                f"{cell['win_count_b']}/{cell['n_scenes']} "
                f"({cell['win_percent_b']:.1f}%) | "
                f"{cell['median_difference_deg']:+.6f} | "
                f"{cell['mean_difference_deg']:+.6f} |"
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--expected-pvalue", type=float, help="optional per-key reference check")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyse_matched_json(args.matched_json, args.expected_pvalue)
    markdown = render_markdown(report)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    args.out_md.write_text(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
