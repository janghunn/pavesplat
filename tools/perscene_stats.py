#!/usr/bin/env python3
"""Main section 4.1: position-paired scene wins, Wilcoxon tests and Holm correction.

Pass baseline then comparison output directories (or comma-separated shard directories
in evaluation-index order). Metric arrays must have exactly matching scene order.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('p-values must be a finite vector in [0, 1]')
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, idx in enumerate(np.argsort(values)):
        running = max(running, (len(values) - rank) * values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def load_run(paths, metrics):
    scenes, scores = [], {metric: [] for metric in metrics}
    for directory in str(paths).split(','):
        path = Path(directory)
        path = path / 'metrics' if (path / 'metrics').is_dir() else path
        keys = json.loads((path / 'scenes_all.json').read_text())
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise ValueError(f'{path}: scenes must be a list of strings')
        for metric in metrics:
            values = np.asarray(json.loads((path / f'scores_{metric}_all.json').read_text()), dtype=float)
            if values.shape != (len(keys),) or not np.isfinite(values).all():
                raise ValueError(f'{path}: {metric} must have one finite value per scene')
            scores[metric].extend(values)
        scenes.extend(keys)
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError('Scenes must be nonempty and unique across shards')
    return scenes, {metric: np.asarray(values) for metric, values in scores.items()}


def paired_runs(run_a, run_b, metrics):
    scenes_a, values_a = load_run(run_a, metrics)
    scenes_b, values_b = load_run(run_b, metrics)
    if scenes_a != scenes_b:
        raise ValueError('Scene order/content differs between runs; cannot position-pair')
    return scenes_a, values_a, values_b


def summarize(run_a, run_b, metrics):
    scenes, first, second = paired_runs(run_a, run_b, metrics)
    rows = []
    for metric in metrics:
        if metric.startswith(('psnr', 'ssim')):
            direction = 1
        elif metric.startswith(('lpips', 'cov_black')):
            direction = -1
        else:
            raise ValueError(f'Unknown improvement direction for {metric}')
        difference = second[metric] - first[metric]
        pvalue = stats.wilcoxon(difference, alternative='two-sided').pvalue if np.any(difference) else 1.0
        rows.append(dict(metric=metric, n=len(scenes), mean_a=float(first[metric].mean()),
                         mean_b=float(second[metric].mean()), mean_difference=float(difference.mean()),
                         win_fraction=float(np.mean(direction * difference > 0)),
                         ties=int(np.sum(difference == 0)), p_wilcoxon=float(pvalue)))
    for row, pvalue in zip(rows, holm([r['p_wilcoxon'] for r in rows])):
        row['p_holm'] = float(pvalue)
    return rows


def write_results(rows, outdir, stem):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (outdir / f'{stem}.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |']
    lines += ['| ' + ' | '.join(str(row[key]) for key in fields) + ' |' for row in rows]
    markdown = '\n'.join(lines) + '\n'
    (outdir / f'{stem}.md').write_text(markdown)
    print(markdown, end='')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_a', help='baseline directory or comma-separated shard directories')
    parser.add_argument('run_b', help='comparison directory or comma-separated shard directories')
    parser.add_argument('--metrics', default='psnr,ssim,lpips')
    parser.add_argument('--outdir', type=Path, default=Path('results/perscene_stats'))
    args = parser.parse_args(argv)
    metrics = args.metrics.split(',')
    if not all(metrics) or len(set(metrics)) != len(metrics):
        parser.error('--metrics must be a nonempty list of unique metric names')
    write_results(summarize(args.run_a, args.run_b, metrics), args.outdir, 'perscene_stats')


if __name__ == '__main__':
    main()
