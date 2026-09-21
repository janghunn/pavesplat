#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
usage() { cat <<'USAGE'
Usage: scripts/eval_scannet_normals.sh <pavesplat|surfsplat> <checkpoint> [--gpu ID] [--npz] [--images] [--dry-run]
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
method=$1; checkpoint=$2; shift 2
[[ "$method" == pavesplat || "$method" == surfsplat ]] || { usage >&2; exit 2; }
gpu=${CUDA_VISIBLE_DEVICES-0}
gpu=${gpu%%,*}
gpu_explicit=false
extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; gpu=$2; gpu_explicit=true; shift ;;
    --npz) extra+=(test.save_normal_npz=true) ;;
    --images) extra+=(test.save_image=true test.save_gt_image=true test.save_input_images=true test.save_depth=true 'test.save_scene_names=[all]') ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
[[ -n "$gpu" && "$gpu" != *,* && "$gpu" != -* && "$gpu" != *[[:space:]]* ]] || { echo '--gpu must select one GPU id.' >&2; exit 2; }
if ! "$gpu_explicit" && [[ ${CUDA_VISIBLE_DEVICES-} == *,* ]]; then
  printf 'Evaluation runs on a single GPU; using first listed id %s from CUDA_VISIBLE_DEVICES.\n' "$gpu" >&2
fi
export CUDA_VISIBLE_DEVICES="$gpu"
cmd=(python -m src.main "+experiment=zeroshot_scannet_${method}_vitl" mode=test
     dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_scannet.json
     checkpointing.resume=false "checkpointing.load=$checkpoint" wandb.mode=disabled
     "output_dir=outputs/eval/scannet_normals_$method" test.eval_normal_error=true "${extra[@]}")
if "$dry_run"; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${cmd[@]}"; printf '\n'
else
  exec "${cmd[@]}"
fi
