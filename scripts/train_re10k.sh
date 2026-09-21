#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Paper runs used 2 GPUs (16×2×1 or 4×2×4); no BatchNorm, so per-GPU batch affects memory.
# Effective batch is 32 with 150K optimizer steps.
usage() { cat <<'USAGE'
Usage: scripts/train_re10k.sh <experiment> [--gpus N] [--dry-run] [Hydra overrides...]
Train a RE10K paper experiment at effective batch 32 / 150K steps.
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
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
experiment=$1; shift
[[ -f "config/experiment/$experiment.yaml" && "$experiment" == *_re10k ]] || { echo "Unknown RE10K experiment: $experiment" >&2; exit 2; }
vit=""
for size in vits vitb vitl; do [[ "$experiment" != *"$size"* ]] || vit=$size; done
[[ -n "$vit" ]] || { echo 'Experiment must identify a backbone.' >&2; exit 2; }
visible=()
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
  IFS=',' read -r -a visible <<< "$CUDA_VISIBLE_DEVICES"
  [[ ${#visible[@]} -gt 0 && "$CUDA_VISIBLE_DEVICES" != -1 ]] || { echo 'No visible GPUs.' >&2; exit 2; }
fi
count=${#visible[@]}
[[ $count -gt 0 ]] || count=1
extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; count=$2; shift 2 ;;
    data_loader.train.batch_size=*|trainer.accumulate_grad_batches=*|trainer.num_nodes=*) echo 'Batch and GPU settings are managed by this launcher.' >&2; exit 2 ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) extra+=("$1"); shift ;;
  esac
done
[[ "$count" =~ ^[1-9][0-9]*$ ]] && (( count <= 8 && 32 % (4 * count) == 0 )) || { echo 'GPU count must be 1, 2, 4, or 8 for effective batch 32.' >&2; exit 2; }
if [[ ${#visible[@]} -eq 0 ]]; then
  for ((i=0; i<count; i++)); do visible+=("$i"); done
elif [[ ${#visible[@]} -ne $count ]]; then
  echo '--gpus must match the number of entries in CUDA_VISIBLE_DEVICES.' >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${visible[*]}")
acc=$((32 / (4 * count)))
cmd=(python -m src.main "+experiment=$experiment" data_loader.train.batch_size=4
     "trainer.accumulate_grad_batches=$acc"
     "checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_$vit.pth"
     checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth
     "output_dir=checkpoints/$experiment" "${extra[@]}")
if "$dry_run"; then printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"; fi
if "$dry_run"; then
  printf '%q ' "${cmd[@]}"; printf '\n'
else
  exec "${cmd[@]}"
fi
