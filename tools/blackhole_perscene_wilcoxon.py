#!/usr/bin/env python3
"""Rebuttal black-hole analysis: adjacent paired runs, with Holm across that family.

Each --runs entry is name=dir[,dir...]; list shards in evaluation-index order.
Deltas are comparison minus preceding run; lower black-hole fraction is better.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import stats

try:
    from .perscene_stats import holm, load_run, write_results
except ImportError:
    from perscene_stats import holm, load_run, write_results


def paired_stats(from_cfg, to_cfg, scores):
    """Replicate the paper's paired per-scene statistics for one pair."""
    diff = scores[to_cfg] - scores[from_cfg]
    n = len(diff)
    mean = float(diff.mean())
    std = float(diff.std(ddof=1))
    se = std / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    direction = -1  # Black-hole fraction is lower-is-better.
    better = int(np.sum(direction * diff > 0))
    ties = int(np.sum(diff == 0))
    p_w = (
        stats.wilcoxon(diff, alternative="two-sided").pvalue
        if np.any(diff)
        else 1.0
    )
    p_s = (
        stats.binomtest(better, n - ties, 0.5).pvalue
        if n - ties
        else 1.0
    )

    nonzero = diff[diff != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    rank_mass = w_plus + w_minus
    r_rb = (w_minus - w_plus) / rank_mass if rank_mass else 0.0

    return {
        "n": n,
        "n_ties": ties,
        "mean_diff_pct": mean * 100,
        "ci_lo_pct": float(mean - tcrit * se) * 100,
        "ci_hi_pct": float(mean + tcrit * se) * 100,
        "dz": mean / std if std > 0 else float("inf"),
        "median_diff_pct": float(np.median(diff)) * 100,
        "win_frac": better / n,
        "p_wilcoxon": float(p_w),
        "p_sign": float(p_s),
        "r_rb": r_rb,
    }


def compare(runs, metric='cov_black_s8'):
    names, scores, reference = [], {}, None
    for specification in runs:
        name, separator, directories = specification.partition('=')
        if not separator or not name or not directories or name in scores:
            raise ValueError('Each run must have a unique name=directory[,directory...]')
        scenes, arrays = load_run(directories, [metric])
        if reference is not None and reference != scenes:
            raise ValueError('Scene order/content differs between runs; cannot position-pair')
        reference = scenes
        names.append(name)
        scores[name] = arrays[metric]
        if np.any((scores[name] < 0) | (scores[name] > 1)):
            raise ValueError('Black-hole fractions must lie in [0, 1]')
    if len(names) < 2:
        raise ValueError('At least two runs are required')
    rows = []
    for first, second in zip(names, names[1:]):
        delta = (scores[second] - scores[first]) * 100
        tail_n = int(0.01 * len(delta))
        tail = np.sort(delta)[::-1][:tail_n]
        rows.append(dict(from_run=first, to_run=second, metric=metric,
                         mean_from_pct=float(scores[first].mean()) * 100,
                         mean_to_pct=float(scores[second].mean()) * 100,
                         median_from_pct=float(np.median(scores[first])) * 100,
                         median_to_pct=float(np.median(scores[second])) * 100,
                         **paired_stats(first, second, scores),
                         worst_one_percent_n=tail_n, worst_one_percent_sum_pp=float(tail.sum())))
    for row, pvalue in zip(rows, holm([row['p_wilcoxon'] for row in rows])):
        row['p_holm'] = float(pvalue)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--metric', default='cov_black_s8')
    parser.add_argument('--outdir', type=Path, default=Path('results/blackhole_stats'))
    args = parser.parse_args(argv)
    write_results(compare(args.runs, args.metric), args.outdir, 'blackhole_perscene_wilcoxon')


if __name__ == '__main__':
    main()
