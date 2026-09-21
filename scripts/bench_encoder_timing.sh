#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Supplement runtime: alternate methods to limit GPU clock and thermal drift.
usage() { cat <<'USAGE'
Usage: scripts/bench_encoder_timing.sh <checkpoint_pavesplat> <checkpoint_surfsplat> [gpu] [--dry-run]
Run three alternating pairs, then summarize paired median encoder overhead.
USAGE
}
for arg in "$@"; do
  if [[ "$arg" == -h || "$arg" == --help ]]; then usage; exit 0; fi
done
dry_run=false
args=()
for arg in "$@"; do
  if [[ "$arg" == --dry-run ]]; then dry_run=true; else args+=("$arg"); fi
done
set -- "${args[@]}"
[[ $# -ge 2 && $# -le 3 ]] || { usage >&2; exit 2; }
checkpoint_pavesplat=$1; checkpoint_surfsplat=$2; gpu=${3-0}
[[ -n "$gpu" && "$gpu" != *,* && "$gpu" != -* && "$gpu" != *[[:space:]]* ]] || { echo 'gpu must select one GPU id.' >&2; exit 2; }
root=outputs/encoder_timing
if ! "$dry_run" && [[ -e "$root" ]]; then
  echo "Refusing to reuse existing timing directory: $root" >&2; exit 2
fi
for pair in 1 2 3; do
  for method in pavesplat surfsplat; do
    checkpoint=$checkpoint_pavesplat
    [[ "$method" != surfsplat ]] || checkpoint=$checkpoint_surfsplat
    out="$root/${method}_$pair"
    cmd=(env "CUDA_VISIBLE_DEVICES=$gpu" PAVESPLAT_SYNC_TIMING=1 python -m src.main
         "+experiment=${method}_vitl_re10k" mode=test dataset/view_sampler=evaluation
         dataset.view_sampler.index_path=assets/evaluation_index_re10k.json
         dataset.test_chunk_interval=200 test.eval_time_skip_steps=10 data_loader.test.num_workers=4
         "output_dir=$out" checkpointing.resume=false "checkpointing.load=$checkpoint" wandb.mode=disabled)
    if "$dry_run"; then
      printf '%q ' "${cmd[@]}"; printf '\n'
    else
      mkdir -p "$out"
      "${cmd[@]}" > "$out/run.log" 2>&1
    fi
  done
done
if "$dry_run"; then exit 0; fi
python - "$root" <<'SUMMARY'
import json
from pathlib import Path
import statistics as stats
import sys
root = Path(sys.argv[1])
def median(method, pair):
    paths = list((root / f'{method}_{pair}').rglob('benchmark.json'))
    if len(paths) != 1:
        raise ValueError(f'Expected one benchmark for {method} pair {pair}, found {len(paths)}')
    values = json.loads(paths[0].read_text())['encoder']
    body = values[10:] if len(values) > 20 else values
    return stats.median(body)
pairs = []
for pair in (1, 2, 3):
    method, baseline = median('pavesplat', pair), median('surfsplat', pair)
    overhead = 100 * (method - baseline) / baseline
    pairs.append((method, baseline, overhead))
    print(f'pair{pair}: PaveSplat={1000*method:.2f} ms SurfSplat={1000*baseline:.2f} ms overhead={overhead:+.2f}%')
print(f'Median paired overhead: {stats.median(p[2] for p in pairs):+.2f}%')
print(f'Mean paired overhead: {stats.mean(p[2] for p in pairs):+.2f}%')
SUMMARY
