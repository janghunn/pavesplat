# Checkpoint compatibility

Two ViT-L checkpoints are selected for release in the [Hugging Face model repository](https://huggingface.co/janghunn/pavesplat). Download [PaveSplat ViT-L](https://huggingface.co/janghunn/pavesplat/resolve/main/pavesplat_vitl_re10k.ckpt), [SurfSplat‡ ViT-L](https://huggingface.co/janghunn/pavesplat/resolve/main/surfsplat_vitl_re10k.ckpt), and [SHA256SUMS](https://huggingface.co/janghunn/pavesplat/resolve/main/SHA256SUMS). The tables below identify the verified raw Lightning files and weights-only release files, without bundling weights in the code repository.

| Model / paper rows | Experiment | Raw file | Size (bytes) | SHA256 | State dictionary |
|---|---|---|---:|---|---|
| PaveSplat ViT-L; main Tables 1–4, supplement Tables 1–4 / Figures 2–6, rebuttal Tables A–B | `pavesplat_vitl_re10k` | `re10k_core_l_step150000.ckpt` | 4253234422 | `e217f1b34f9c70e27cbea7eac73098e71b776c0025cef7db298211387bc51387` | 919 tensors; 36-channel head; camera-space frame, tangent frame, residual rotation enabled |
| SurfSplat‡ ViT-L; controlled baseline in the same tables/figures | `surfsplat_vitl_re10k` | `re10k_baseline_l_step150000.ckpt` | 4253120438 | `de3301a1214528a88e69d142777ad8325aa7deaf5fcec001320cb5cdf1725494` | 919 tensors; 32-channel head; all three adapter flags disabled |

## Weights-only release files

| File | Size (bytes) | SHA256 | Tensors | Parameters | Head channels |
|---|---:|---|---:|---:|---:|
| `pavesplat_vitl_re10k.ckpt` | 1417766474 | `3dc6819691b87528b5f39cf3923e6e883ece08908bbccf206db4e701e5908bea` | 919 | 354,335,337 | 36 |
| `surfsplat_vitl_re10k.ckpt` | 1417728458 | `f864ee61c97903113926a9cf96638ec598ffd10ff119faa263337f5825822e93` | 919 | 354,325,825 | 32 |

These files retain `state_dict`, `epoch`, `global_step`, and `pytorch-lightning_version`; optimizer, scheduler, and loop state are stripped. Both files load with `checkpointing.load` without any key remap. The parameter difference is 9,512, matching the residual head channels reported in the paper.

Both use ViT-L, two matching scales, upsampling factor 2, one Gaussian per pixel and spherical-harmonic degree 2. Keys keep the original `encoder.depth_predictor.*`, `encoder.feature_upsampler.*`, `encoder.gaussian_regressor.*`, and `encoder.gaussian_head.*` names. No key remapping is needed. The config is supplied separately; it is not embedded as model hyperparameters. CPU key/shape equality is tested by `tests/test_checkpoint_keys.py` when `PAVESPLAT_CKPT_DIR` points to these raw files.

Training: RE10K 256×256, effective batch 32, 150K optimizer steps, MSE + 0.05 LPIPS from step 0, learning rates 0.0002 / 0.000002 (monodepth), opacity cap 0.6, alpha thresholds 0.1 during training / 0.001 during evaluation. The two-GPU YAML recipe is batch 4 with accumulation 4; `scripts/train_re10k.sh` adapts accumulation to GPU count.

The designated checkpoint license is **CC-BY-NC-4.0**, reflecting the [Depth Anything V2 ViT-L weight lineage](https://github.com/DepthAnything/Depth-Anything-V2#license). Code and rasterizer terms are listed separately in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

| Model | Config | Raw size (bytes) | Availability / paper row |
|---|---|---:|---|
| PaveSplat ViT-S | `pavesplat_vits_re10k` | 460884982 | Not released; config provided. The retrained checkpoint gives 27.107 dB, printed in camera-ready supplement Table 4 and rebuttal Table B; the original checkpoint behind the submission-version 27.100 dB is unavailable. |
| SurfSplat‡ ViT-S | `surfsplat_vits_re10k` | 460798582 | Not released; config provided. Supplement Table 4 / rebuttal Table B. |
| PaveSplat ViT-B | `pavesplat_vitb_re10k` | 1449943958 | Not released; config provided. Supplement Table 4 / rebuttal Table B. |
| + tangent frame, ViT-L | `ablation_tangent_frame_vitl_re10k` | 4253120438 | Not released; config provided. Main Table 4 / rebuttal black-hole analysis. |
| SurfSplat ViT-B | External | — | Cited paper value; no same-recipe checkpoint was trained. |
| ScanNet in-domain | — | — | Unreported; no checkpoint released. ScanNet experiments use RE10K-trained ViT-L weights zero-shot. |

Host locations are recorded in the release audit’s `evidence/checkpoint-inventory.md`, outside this repository.
