#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
usage() { cat <<'USAGE'
Usage: scripts/eval_re10k.sh <experiment> <checkpoint> [--gpu ID] [--8x] [--coverage] [--images] [--timing] [--dry-run] [Hydra overrides...]
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
[[ $# -ge 2 ]] || { usage >&2; exit 2; }
experiment=$1; checkpoint=$2; shift 2
[[ -f "config/experiment/$experiment.yaml" && "$experiment" == *_re10k ]] || { echo "Unknown RE10K experiment: $experiment" >&2; exit 2; }
out="outputs/eval/$experiment"
gpu=${CUDA_VISIBLE_DEVICES-0}
gpu=${gpu%%,*}
gpu_explicit=false
extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; gpu=$2; gpu_explicit=true; shift ;;
    --8x) out="outputs/eval/${experiment}_8x"; extra+=(test.hrrc_eval_8x_only=true) ;;
    --coverage) extra+=(test.eval_coverage=true) ;;
    --images) extra+=(test.save_image=true test.save_gt_image=true test.save_input_images=true 'test.save_scene_names=[all]') ;;
    --timing) export PAVESPLAT_SYNC_TIMING=1; extra+=(test.eval_time_skip_steps=10) ;;
    *=*) extra+=("$1") ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
[[ -n "$gpu" && "$gpu" != *,* && "$gpu" != -* && "$gpu" != *[[:space:]]* ]] || { echo '--gpu must select one GPU id.' >&2; exit 2; }
if ! "$gpu_explicit" && [[ ${CUDA_VISIBLE_DEVICES-} == *,* ]]; then
  printf 'Evaluation runs on a single GPU; using first listed id %s from CUDA_VISIBLE_DEVICES.\n' "$gpu" >&2
fi
export CUDA_VISIBLE_DEVICES="$gpu"
cmd=(python -m src.main "+experiment=$experiment" mode=test dataset/view_sampler=evaluation
     dataset.view_sampler.index_path=assets/evaluation_index_re10k.json
     checkpointing.resume=false "checkpointing.load=$checkpoint" wandb.mode=disabled
     "output_dir=$out" "${extra[@]}")
if "$dry_run"; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  if [[ ${PAVESPLAT_SYNC_TIMING:-0} == 1 ]]; then printf 'PAVESPLAT_SYNC_TIMING=1 '; fi
  printf '%q ' "${cmd[@]}"; printf '\n'
else
  exec "${cmd[@]}"
fi
