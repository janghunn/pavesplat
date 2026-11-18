# PaveSplat: Resolution-Robust Feed-Forward Gaussian Splatting with Residually Corrected Surface-Aligned Covariances

**ACCV 2026** · Janghun Kim, Hyun-Cheol Kim, Eunbin An, Myeonghun Oh, Hyon-Gon Choo

ETRI, South Korea

PaveSplat learns a residual reorientation Δq of a surface-aligned covariance prior, maintaining surface paving when Gaussian splats are rendered at higher resolutions.

[project page](https://janghunn.github.io/pavesplat/) · [code](https://github.com/janghunn/pavesplat) · [checkpoints](https://huggingface.co/janghunn/pavesplat)

The coordinate-consistent formulation constructs the tangent-aligned frame `R_a=[e1 e2 n]` and applies `q = q_a ⊗ Δq` (paper Eqs. 2–6; supplement Algorithm 1). The residual adds 9,512 head parameters. The released SurfSplat‡ reproduction retains its world-space branch and shares the backbone, losses, opacity cap, alpha thresholds and rasterizer.

## Installation

The tested stack is Python 3.12, PyTorch 2.7.1+cu128, Lightning 2.4.0, CUDA 12.8 toolkit and GCC 13 on Blackwell. The upstream PyTorch 2.4/CUDA 12.4 stack works on older GPUs; the pinned requirements here describe the tested stack. Set `TORCH_CUDA_ARCH_LIST` for your GPU (12.0 below is Blackwell). Clone the code and submodule, then run the installation commands from the repository root in your Python environment:

```bash
git clone --recursive https://github.com/janghunn/pavesplat.git # not executed here
cd pavesplat # not executed here
```

For an existing checkout, initialize its submodules with the first command below. Before building the rasterizer on CUDA 12.8 / GCC 13, apply the supplied compatibility patch. The `--check` guard makes application skip-safe when the patch is already applied: the check returns nonzero and `&&` skips the second command. Older toolchains may not need it. The patch adds only `<cstdint>` includes to two headers and makes no algorithm change.

```bash
git submodule update --init --recursive # not executed here
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 # not executed here
python -m pip install -r requirements.txt # not executed here
git -C submodules/rasterization apply --check ../../patches/rasterization-cstdint.patch && git -C submodules/rasterization apply ../../patches/rasterization-cstdint.patch
TORCH_CUDA_ARCH_LIST=12.0 CC=gcc-13 CXX=g++-13 python -m pip install --no-build-isolation submodules/rasterization # not executed here
```

The ellipsoid figure additionally needs [nvdiffrast](https://github.com/NVlabs/nvdiffrast); it is optional for training and image evaluation. Cloning, patch application, installation and full GPU/data runs above and below were not executed here. The documentation gate executes only the patch's `--check` command; launcher dry-runs, config composition, tool help and CPU tests validate the remaining commands.

## Data

Use [pixelSplat’s RE10K/ACID chunks and preparation instructions](https://github.com/dcharatan/pixelsplat). Place or symlink dataset roots at `datasets/re10k`, `datasets/acid`, `datasets/dl3dv`, and `datasets/scannet`; test chunks and `index.json` belong under each root’s `test/` directory. RE10K training also needs `train/`. Shipped evaluation indices are frozen; regenerating RE10K/ACID pairs produces a different random selection.

Use the retained converters for DL3DV test data and raw ScanNet frames:

```bash
python src/scripts/convert_dl3dv_test.py --input_dir datasets/dl3dv_raw --output_dir datasets/dl3dv --img_subdir images_8 # not executed here
python src/scripts/generate_chunk_index.py --root datasets/dl3dv/test # not executed here
python src/scripts/convert_scan.py --input_dir datasets/scannet_raw/test --output_dir datasets/scannet --stage test # not executed here
python src/scripts/generate_scannet_eval_index.py --raw_test datasets/scannet_raw/test --out assets/evaluation_index_scannet.json # not executed here
```

These conversion commands require data and were not executed here. ScanNet’s SurfSplat split is unreleased: our zero-shot protocol uses its slicing convention and the repository’s overlap geometry with seed 0, yielding 635 slices (633 with GT depth), as described in supplement section 3. Nominal slices have 350 frames and 20-frame overlap; the converter merges a remaining tail shorter than 100 frames. Both methods consume the same frozen index. ScanNet geometry results refer to **rendered-normal error**.

DL3DV follows the paper’s unified **256×256** protocol, retains the original 270×480 shape check, uses the 2-context/4-target index and near/far 0.5/200, and drops one scene with inconsistent frame dimensions (`skip_bad_shape=true`), leaving **139 scenes**.

**Dataset terms:** No dataset is redistributed. Obtain each dataset from its official source and accept its terms: [RealEstate10K](https://google.github.io/realestate10k/) camera data is CC BY 4.0 (Google LLC); for [ACID](https://infinite-nature.github.io/), follow the terms on the official page; [DL3DV-10K](https://github.com/DL3DV-10K/Dataset) is under the DL3DV-10K Terms of Use and CC BY-NC 4.0; [ScanNet](http://www.scan-net.org/) is under the ScanNet Terms of Use.

## Pretrained weights

For training, download [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) weights to `pretrained/depth_anything_v2_vits.pth`, `pretrained/depth_anything_v2_vitb.pth` or `pretrained/depth_anything_v2_vitl.pth`, matching the experiment. Download [GMFlow](https://github.com/haofeixu/gmflow) weights as `pretrained/gmflow-scale1-things-e9887eda.pth`. DINOv2 is loaded through `torch.hub` on first model construction; allow the initial download or populate its cache in advance.

## Checkpoints

| Model | Experiment | Download |
|---|---|---|
| PaveSplat ViT-L | `pavesplat_vitl_re10k` | [Weights](https://huggingface.co/janghunn/pavesplat/resolve/main/pavesplat_vitl_re10k.ckpt) |
| SurfSplat‡ ViT-L | `surfsplat_vitl_re10k` | [Weights](https://huggingface.co/janghunn/pavesplat/resolve/main/surfsplat_vitl_re10k.ckpt) |

[Checkpoint identities, hashes and unavailable rows](docs/CHECKPOINTS.md) describe the two selected checkpoints; [SHA256SUMS](https://huggingface.co/janghunn/pavesplat/resolve/main/SHA256SUMS) lists the release hashes. Examples use local aliases `checkpoints/pavesplat_vitl.ckpt` and `checkpoints/surfsplat_vitl.ckpt`. Download access follows the model repository's visibility; no weights are bundled in the code repository. ViT-S, ViT-B and tangent-frame ablation checkpoints are not released.

## Training

```bash
scripts/train_re10k.sh pavesplat_vitl_re10k --gpus 2
scripts/train_re10k.sh surfsplat_vitl_re10k --gpus 2
python scripts/check_configs.py
```

The paper uses effective batch 32 and 150K optimizer steps (`max_steps=150001`). With per-GPU batch 4 the launcher sets accumulation to `32/(4×N)` and accepts 1, 2, 4 or 8 GPUs. If `CUDA_VISIBLE_DEVICES` is set, its entry count is the default and must match `--gpus`; otherwise one GPU is the default. The paper used two GPUs (16×2×1 or 4×2×4). Training has no BatchNorm. W&B defaults to online mode, project `pavesplat`, and a null entity; pass `wandb.mode=disabled` to disable logging. Training was not executed here.

## Evaluation

Evaluation always runs on a single GPU; pass `--gpu N` to choose it.

```bash
scripts/eval_re10k.sh pavesplat_vitl_re10k checkpoints/pavesplat_vitl.ckpt
scripts/eval_re10k.sh surfsplat_vitl_re10k checkpoints/surfsplat_vitl.ckpt
scripts/eval_re10k.sh pavesplat_vitl_re10k checkpoints/pavesplat_vitl.ckpt --8x --coverage --images
scripts/eval_zeroshot.sh dl3dv pavesplat checkpoints/pavesplat_vitl.ckpt --8x --coverage
scripts/eval_zeroshot.sh acid pavesplat checkpoints/pavesplat_vitl.ckpt
scripts/eval_zeroshot.sh scannet pavesplat checkpoints/pavesplat_vitl.ckpt
scripts/eval_scannet_normals.sh pavesplat checkpoints/pavesplat_vitl.ckpt --npz --images
scripts/bench_encoder_timing.sh checkpoints/pavesplat_vitl.ckpt checkpoints/surfsplat_vitl.ckpt 0
```

Full evaluations require a CUDA GPU, data and the matching checkpoint and were not executed here. Each shell launcher accepts `-h` and `--dry-run`. Evaluation explicitly loads the given checkpoint. Standard HRRC renders 1×/2×/4× with bicubic GT; `--8x` renders the separate 1×/8× pass and scores 8×. Metric suffixes are none/2/3/4 for 1×/2×/4×/8×. Results are written to `outputs/eval/<experiment>[_8x]/metrics/`, including `scores_all_avg.json`, per-scene arrays and `scenes_all.json`.

`--coverage` records accumulated-alpha coverage and black-hole fraction: rendered max-RGB < ε where GT max-RGB > ε (max RGB is the luma proxy), with ε = 0.01/0.02/0.05. Timing uses three alternating pairs and `PAVESPLAT_SYNC_TIMING=1`; its launcher refuses an existing `outputs/encoder_timing/` directory.

## Reproduce

[REPRODUCE.md](docs/REPRODUCE.md) maps every retained main-paper, supplement and rebuttal artifact to configs, exact commands, outputs and checkpoint requirements. [CONFIG_NAMES.md](docs/CONFIG_NAMES.md) maps the released experiment names and shared configuration groups to their paper rows.

## Statistics tools

```bash
python tools/perscene_stats.py outputs/eval/surfsplat_vitl_re10k outputs/eval/pavesplat_vitl_re10k --metrics psnr,ssim,lpips --outdir results/perscene
python tools/paired_stats.py results/pavesplat.wandb results/surfsplat.wandb --outdir results/paired
python tools/compare_runs.py results/pavesplat.wandb results/surfsplat.wandb
python tools/matched_normal_eval.py outputs/eval/scannet_normals_surfsplat outputs/eval/scannet_normals_pavesplat --index assets/evaluation_index_scannet.json --name-a SurfSplat --name-b PaveSplat --out-json results/matched.json --out-md results/matched.md
python tools/scene_level_wilcoxon.py --matched-json results/matched.json --out-json results/scene_normals.json --out-md results/scene_normals.md
python tools/blackhole_perscene_wilcoxon.py --runs surfsplat=outputs/eval/surfsplat_vitl_re10k_8x pavesplat=outputs/eval/pavesplat_vitl_re10k_8x --outdir results/blackholes
```

These tools accept `--help`; the data-dependent analyses were not executed here. Per-scene tools validate identical scene order and accept comma-separated shard directories in index order. Win fractions include ties in the denominator; Wilcoxon tests are two-sided and Holm correction is applied within the reported metric or adjacent-pair family. Matched normals use the common valid pixels, then aggregate slices by physical scene for rebuttal Table A.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{kim2026pavesplat,
  title     = {PaveSplat: Resolution-Robust Feed-Forward Gaussian Splatting with Residually Corrected Surface-Aligned Covariances},
  author    = {Janghun Kim and Hyun-Cheol Kim and Eunbin An and Myeonghun Oh and Hyon-Gon Choo},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## Acknowledgements

Built on [SurfSplat](https://github.com/hebing-sjtu/SurfSplat), [DepthSplat](https://github.com/cvg/depthsplat), [MVSplat](https://github.com/donydchen/mvsplat), [pixelSplat](https://github.com/dcharatan/pixelsplat), [UniMatch](https://github.com/autonomousvision/unimatch), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [DINOv2](https://github.com/facebookresearch/dinov2), and the [upstream SurfSplat variant of the Inria/MPII Gaussian Splatting rasterizer](https://github.com/hebing-sjtu/rasterization-w-alpha-depth). PaveSplat modifications add correctable surface-aligned covariances, residual initialization and diagnostics, HRRC/coverage evaluation and the documented release tools. Upstream method branches and attribution remain preserved.

## License

Code is under [CC BY-NC-SA 4.0](LICENSE). The rasterizer retains its [research-only Gaussian-Splatting license](submodules/rasterization/LICENSE.md). Selected ViT-L checkpoint weights are designated CC-BY-NC-4.0 because of their Depth Anything V2 lineage. See [full third-party notices](THIRD_PARTY_NOTICES.md); individual components and external weights retain their own terms. Dataset terms are listed under [Data](#data).
