#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
usage() { cat <<'USAGE'
Usage: scripts/eval_zeroshot.sh <dl3dv|acid|scannet> <pavesplat|surfsplat> <checkpoint> [--gpu ID] [--8x] [--coverage] [--dry-run] [Hydra overrides...]
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
[[ $# -ge 3 ]] || { usage >&2; exit 2; }
dataset=$1; method=$2; checkpoint=$3; shift 3
case "$dataset" in
  dl3dv) index=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json ;;
  acid|scannet) index="assets/evaluation_index_$dataset.json" ;;
  *) usage >&2; exit 2 ;;
esac
[[ "$method" == pavesplat || "$method" == surfsplat ]] || { usage >&2; exit 2; }
experiment="zeroshot_${dataset}_${method}_vitl"
out="outputs/eval/$experiment"
gpu=${CUDA_VISIBLE_DEVICES-0}
gpu=${gpu%%,*}
gpu_explicit=false
extra=()
[[ "$dataset" != dl3dv ]] || extra+=(dataset.view_sampler.num_context_views=2)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; gpu=$2; gpu_explicit=true; shift ;;
    --8x) out="outputs/eval/${experiment}_8x"; extra+=(test.hrrc_eval_8x_only=true) ;;
    --coverage) extra+=(test.eval_coverage=true) ;;
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
     "dataset.view_sampler.index_path=$index" checkpointing.resume=false "checkpointing.load=$checkpoint"
     wandb.mode=disabled "output_dir=$out" "${extra[@]}")
if "$dry_run"; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${cmd[@]}"; printf '\n'
else
  exec "${cmd[@]}"
fi
