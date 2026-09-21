from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat, einsum
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import hashlib
import json
import os
import time
from tqdm import tqdm
import torch.nn.functional as F
import math

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim, compute_depth_errors, compute_depth_errors_aligned
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video, save_raw_depth
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode, DecoderOutput
from .types import Gaussians
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from src.visualization.vis_depth import viz_depth_tensor, viz_normal_tensor
from PIL import Image
from ..misc.stablize_camera import render_stabilization_path
from .ply_export import save_gaussian_ply


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    lr_monodepth: float
    weight_decay: float


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    save_gt_image: bool
    save_input_images: bool
    save_depth: bool
    save_depth_concat_img: bool
    save_depth_npy: bool
    save_gaussian: bool
    render_chunk_size: int | None
    stablize_camera: bool
    stab_camera_kernel: int
    # Rendered-normal evaluation, disabled by default.
    eval_normal_error: bool = False    # angular error of rendered normal vs GT-depth normal
    save_normal_npz: bool = False      # persist matched-pixel normal evidence (requires normal eval)
    save_eval_plots: bool = False      # dump the normal-error histogram (lazy matplotlib)
    eval_min_depth: float = 0.5        # GT depth mask lower bound (matches compute_depth_errors)
    eval_max_depth: float = 10.0       # GT depth mask upper bound
    normal_edge_rel_thresh: float = 0.05  # rel inverse-depth jump that excludes a GT-normal pixel
    normal_min_norm: float = 0.1       # composited-normal coverage floor (drop unhit pixels)
    # HRRC 8x (2048^2) re-render — dedicated pass, eval-only, default OFF. The standard test
    # pass already scores 1x/2x/4x (psnr/psnr2/psnr3); SurfSplat's HRRC table stops at 4x. To
    # extend the merged multi-resolution Table 1 to 8x WITHOUT redundantly re-rendering 1x/2x/4x,
    # set this True: the test_step then renders ONLY 1x (structurally required) + 8x and writes
    # ONLY the 2048^2 scores as psnr4/ssim4/lpips4 (suffix convention: none=1x,2=2x,3=4x,4=8x).
    # The expensive 2x (512^2) and 4x (1024^2) decoder passes are skipped. Off => byte-identical.
    hrrc_eval_8x_only: bool = False
    # Scene filter for save_image/save_input_images. None = legacy hardcoded
    # test_scene_names list (upstream behavior, byte-identical); ["all"] = save
    # every evaluated scene (use with a reduced evaluation index!); otherwise an
    # explicit list of scene names.
    save_scene_names: list[str] | None = None
    # Mean alpha and C_tau = P(alpha >= tau) at each HRRC render scale.
    # Black-hole fractions use rendered and GT max-channel RGB thresholds.
    eval_coverage: bool = False
    coverage_taus: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.5])
    black_eps_sweep: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])
    # --- Ellipsoid-soup figure (each Gaussian drawn as an oriented covariance ellipsoid
    #     instead of a splat, to show surface-paving / hole-filling; eval-only, default OFF
    #     so existing runs stay byte-identical) ---
    save_ellipsoids: bool = False
    ellipsoid_sigmas: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    ellipsoid_color_mode: str = "scene"   # "scene" (SH DC albedo) | "flat" | "normal"
    ellipsoid_opacity_min: float = 0.05   # drop near-invisible Gaussians before drawing
    ellipsoid_scale: int = 8              # render-res multiplier of the target view (matches 8x)
    ellipsoid_bg: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    eval_model_every_n_val: int
    eval_data_length: int
    eval_deterministic: bool
    eval_time_skip_steps: int
    eval_save_model: bool
    l1_loss: bool
    intermediate_loss_weight: float
    no_viz_video: bool
    viz_depth: bool
    forward_depth_only: bool
    train_ignore_large_loss: float
    no_log_projections: bool
    # HRRC (high-resolution re-rendering consistency) live val monitoring — eval-only, default OFF
    # so existing runs are byte-identical. Re-renders the val view at 2x/4x to log PSNR/SSIM/LPIPS
    # drop curves + a 1x/2x/4x panel. hrrc_val_scenes caps the per-epoch cost.
    hrrc_val: bool = False
    hrrc_val_scenes: int = 32
    # Bicubic GT upsampling follows the SurfSplat HRRC protocol (Eq. 13).
    hrrc_gt_mode: str = "bicubic"


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass

def upsample_gt_img(gt_img, scale, mode="bilinear"):
    """
    gt_img: [B, V, C, H, W] / [V, C, H, W] / [C, H, W]
    target_hw: (H_new, W_new)
    mode: 'bilinear', 'bicubic', 'nearest' (bicubic may overshoot [0,1] — clamp at call site)
    """
    orig_shape = gt_img.shape
    h,w =  orig_shape[-2], orig_shape[-1]
    target_hw = (int(h*scale), int(w*scale))
    ndim = gt_img.ndim

    # input dimension[*, C, H, W]
    if ndim == 5:
        gt_img_ = gt_img.view(-1, *gt_img.shape[-3:])  # [B*V, C, H, W]
    elif ndim == 4:
        gt_img_ = gt_img
    elif ndim == 3:
        gt_img_ = gt_img.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported gt_img shape: {gt_img.shape}")

    gt_img_up = F.interpolate(gt_img_, size=target_hw, mode=mode, align_corners=False)

    if ndim == 5:
        B, V = orig_shape[0], orig_shape[1]
        gt_img_up = gt_img_up.view(B, V, *gt_img_up.shape[1:])
    elif ndim == 3:
        gt_img_up = gt_img_up.squeeze(0)

    return gt_img_up


test_scene_names = [
    "0d4de33c6888a754",
    "4c4fa41951e37e78",
    "5a27f8e07fcdff760",
    "5aca87f95a9412c6",
    "29e0bfbad00f0d5e"
]

# test_scene_names = [
#     "0bfdd020cf475b9c68e4b469d1d1a2d0cad303eefe8b78fb2307855afdaac8be",
#     "ff592398657b3dfe94153332861985194a3e3c9d199c4a3a27a0ce4038e81ade"
# ]


def _scene_save_allowed(scene, save_scene_names):
    """test.save_scene_names: None -> legacy hardcoded list; ['all'] -> every
    scene; else explicit allowlist."""
    if save_scene_names is None:
        return scene in test_scene_names
    if "all" in save_scene_names:
        return True
    return scene in save_scene_names


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    eval_data_cfg: Optional[DatasetCfg | None]

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        eval_data_cfg: Optional[DatasetCfg | None] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.eval_data_cfg = eval_data_cfg

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

        # Pool per-scene pixel errors separately from scalar scene scores.
        if getattr(self.test_cfg, "eval_normal_error", False):
            self._normal_buf = {"oriented": [], "unoriented": []}
        if self.test_cfg.save_normal_npz:
            self._normal_npz_scenes = []
        # Coverage diagnostic: per-scene scalars are stored straight in test_step_outputs
        # (keys cov_meanalpha_s{1,2,4,8}, cov_C{30,50,70,90}_s{...}) so they ride the existing
        # on_test_end averaging + scores_*_all.json per-scene dump, position-aligned with
        # scenes_all.json. s2/s4 populate only on the standard pass, s8 only on the 8x-only
        # pass — exactly the split already used for psnr/psnr2/psnr3/psnr4.
        if getattr(self.test_cfg, "eval_coverage", False):
            self._coverage_taus = list(
                getattr(self.test_cfg, "coverage_taus", [0.01, 0.05, 0.1, 0.5])
            )

    @staticmethod
    def _cov_tau_key(tau: float) -> str:
        """Filename-safe key for a coverage threshold, e.g. 0.5 -> 'C50' (no dots)."""
        return f"C{int(round(tau * 100)):02d}"

    def _append_coverage(self, alpha, scale: int) -> None:
        """Record mean alpha and C_tau = P(alpha >= tau) at one HRRC scale.

        Average across all pixels and target views, with one scalar per scene.
        The arrays share the scene order of the image-quality metrics. Chunked
        rendering does not concatenate alpha maps, so coverage is skipped there.
        """
        if not getattr(self.test_cfg, "eval_coverage", False):
            return
        if alpha is None or self.test_cfg.render_chunk_size is not None:
            return
        a = alpha.detach().float()
        vals = {f"cov_meanalpha_s{scale}": a.mean().item()}
        for t in self._coverage_taus:
            vals[f"cov_{self._cov_tau_key(t)}_s{scale}"] = (a >= t).float().mean().item()
        for k, val in vals.items():
            if k not in self.test_step_outputs:
                self.test_step_outputs[k] = []
            self.test_step_outputs[k].append(val)

    def _append_black_frac(self, rendered, gt, scale: int, eps: float = 0.02) -> None:
        """Record black-hole fractions for Fig. 4 and Supp. Tab. 3.

        A hole has rendered luma < eps where GT luma > eps; the implementation
        uses the maximum RGB channel as the luma proxy. Fractions average over
        all pixels and target views, including pixels outside the GT mask in
        the denominator. cov_blackany_s{scale} counts all near-black pixels.
        The default eps is 0.02; black_eps_sweep adds eps01/eps02/eps05 keys
        from the same render, with eps02 equal to cov_black_s{scale}.
        Inputs are [V,3,H,W]. Disabled or chunked evaluation emits no values.
        """
        if not getattr(self.test_cfg, "eval_coverage", False):
            return
        if rendered is None or gt is None or self.test_cfg.render_chunk_size is not None:
            return
        r = rendered.detach().float().amax(dim=-3)   # [V, H, W] max channel = luma proxy
        g = gt.detach().float().amax(dim=-3)
        black_any = (r < eps)
        black_hole = black_any & (g > eps)
        vals = {
            f"cov_black_s{scale}": black_hole.float().mean().item(),
            f"cov_blackany_s{scale}": black_any.float().mean().item(),
        }
        for e in self.test_cfg.black_eps_sweep:
            tag = f"eps{int(round(float(e) * 100)):02d}"
            vals[f"cov_black_s{scale}_{tag}"] = ((r < e) & (g > e)).float().mean().item()
        for k, val in vals.items():
            if k not in self.test_step_outputs:
                self.test_step_outputs[k] = []
            self.test_step_outputs[k].append(val)

    @staticmethod
    def _alpha_view(output) -> "Tensor | None":
        """First-batch alpha map [V, H, W] from a DecoderOutput, or None if absent."""
        if output is None or getattr(output, "alpha", None) is None:
            return None
        return output.alpha[0]

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        gaussians = self.encoder(
            batch["context"], self.global_step, False, scene_names=batch["scene"]
        )

        diagnostics = None
        if isinstance(gaussians, dict):
            pred_depths = gaussians["depths"]
            diagnostics = gaussians.get("diagnostics")
            gaussians = gaussians["gaussians"]

        supervise_intermediate_depth = False
        batch_size = batch["target"]["extrinsics"].size(0)

        if gaussians.means.size(0) != batch["target"]["extrinsics"].size(0):
            supervise_intermediate_depth = True
            assert gaussians.means.size(0) % batch["target"]["extrinsics"].size(0) == 0
            num_depths = gaussians.means.size(0) // batch["target"]["extrinsics"].size(
                0
            )
            # add loss to intermediate depth predictions
            target_extrinsics = torch.cat(
                [batch["target"]["extrinsics"]] * num_depths, dim=0
            )
            target_intrinsics = torch.cat(
                [batch["target"]["intrinsics"]] * num_depths, dim=0
            )
            target_near = torch.cat([batch["target"]["near"]] * num_depths, dim=0)
            target_far = torch.cat([batch["target"]["far"]] * num_depths, dim=0)

            output_all = self.decoder.forward(
                gaussians,
                target_extrinsics,
                target_intrinsics,
                target_near,
                target_far,
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )
            # split
            batch_size = batch["target"]["extrinsics"].size(0)
            # order: intermediate depth, final depth
            output_intermediate = DecoderOutput(
                color=output_all.color[:-batch_size],
                depth=(
                    output_all.depth[:-batch_size]
                    if output_all.depth is not None
                    else None
                ),
                normal=(
                    output_all.normal[:-batch_size]
                    if output_all.normal is not None
                    else None
                ),
            )
            output = DecoderOutput(
                color=output_all.color[-batch_size:],
                depth=(
                    output_all.depth[-batch_size:]
                    if output_all.depth is not None
                    else None
                ),
                normal=(
                    output_all.normal[-batch_size:]
                    if output_all.normal is not None
                    else None
                ),
            )

        else:
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )

        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr", psnr_probabilistic.mean())

        # Log encoder diagnostics, including mean/max residual rotation angle.
        if diagnostics is not None:
            for diag_name, diag_value in diagnostics.items():
                if torch.is_tensor(diag_value) and diag_value.dim() > 0:
                    continue  # skip non-scalar diagnostics (e.g. delta_q_angle_map, eval-only)
                self.log(f"diag/{diag_name}", diag_value)

        # Compute and log loss.
        total_loss = 0

        valid_depth_mask = None

        for loss_fn in self.losses:
            if loss_fn.name == "mse":
                loss = loss_fn.forward(
                    output,
                    batch,
                    gaussians,
                    self.global_step,
                    l1_loss=self.train_cfg.l1_loss,
                    clamp_large_error=self.train_cfg.train_ignore_large_loss,
                    valid_depth_mask=valid_depth_mask,
                )
            else:
                loss = loss_fn.forward(
                    output,
                    batch,
                    gaussians,
                    self.global_step,
                    valid_depth_mask=valid_depth_mask,
                )
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss

        # color loss on intermediate output
        if supervise_intermediate_depth:
            for loss_fn in self.losses:
                if output_intermediate.color.size(0) != batch_size:
                    assert output_intermediate.color.size(0) % batch_size == 0
                    num_intermediate = output_intermediate.color.size(0) // batch_size
                    intermediate_loss = 0
                    for i in range(num_intermediate):
                        curr_output = DecoderOutput(
                            color=output_intermediate.color[
                                (batch_size * i) : (batch_size * (i + 1))
                            ],
                            depth=(
                                output_intermediate.depth[
                                    (batch_size * i) : (batch_size * (i + 1))
                                ]
                                if output_intermediate.depth is not None
                                else None
                            ),
                            normal=(
                                output_intermediate.normal[
                                    (batch_size * i) : (batch_size * (i + 1))
                                ]
                                if output_intermediate.normal is not None
                                else None
                            ),
                        )
                        curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (
                            num_intermediate - i
                        )

                        if loss_fn.name == "mse":
                            loss = loss_fn.forward(
                                curr_output,
                                batch,
                                gaussians,
                                self.global_step,
                                l1_loss=self.train_cfg.l1_loss,
                                clamp_large_error=self.train_cfg.train_ignore_large_loss,
                                valid_depth_mask=valid_depth_mask,
                            )
                        else:
                            loss = loss_fn.forward(
                                curr_output,
                                batch,
                                gaussians,
                                self.global_step,
                                valid_depth_mask=valid_depth_mask,
                            )

                        intermediate_loss = intermediate_loss + curr_loss_weight * loss

                    self.log(f"loss/{loss_fn.name}_intermediate", intermediate_loss)
                    total_loss = total_loss + intermediate_loss
                else:
                    if loss_fn.name == "mse":
                        loss = loss_fn.forward(
                            output_intermediate,
                            batch,
                            gaussians,
                            self.global_step,
                            l1_loss=self.train_cfg.l1_loss,
                            clamp_large_error=self.train_cfg.train_ignore_large_loss,
                            valid_depth_mask=valid_depth_mask,
                        )
                    else:
                        loss = loss_fn.forward(
                            output_intermediate,
                            batch,
                            gaussians,
                            self.global_step,
                            valid_depth_mask=valid_depth_mask,
                        )
                    self.log(f"loss/{loss_fn.name}_intermediate", loss)
                    total_loss = (
                        total_loss + self.train_cfg.intermediate_loss_weight * loss
                    )

        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # checkpoint monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        if self.test_cfg.save_normal_npz:
            requirements = {
                "test.compute_scores=true": self.test_cfg.compute_scores,
                "test.eval_normal_error=true": self.test_cfg.eval_normal_error,
                "train.forward_depth_only=false": not self.train_cfg.forward_depth_only,
                "test.render_chunk_size=null": self.test_cfg.render_chunk_size is None,
            }
            unmet = [name for name, satisfied in requirements.items() if not satisfied]
            if unmet:
                raise RuntimeError(
                    "test.save_normal_npz requires "
                    + ", ".join(requirements)
                    + "; unmet: "
                    + ", ".join(unmet)
                )

        batch: BatchedExample = self.data_shim(batch)        
        b, v, _, _, _ = batch["target"]["image"].shape
        assert b == 1

        pred_depths = None

        # save input views for visualization
        if self.test_cfg.save_input_images:
            (scene,) = batch["scene"]
            self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
            path = Path(get_cfg()["output_dir"])

            input_images = batch["context"]["image"][0]  # [V, 3, H, W]
            index = batch["context"]["index"][0]
            if _scene_save_allowed(scene, self.test_cfg.save_scene_names):
                for idx, color in zip(index, input_images):
                    save_image(color, path / "images" / scene / f"color/input_{idx:0>6}.png")
                    
        _, _, _, h, w = batch["context"]["image"].shape
        
        # save depth vis
        if self.test_cfg.save_depth or self.test_cfg.save_gaussian:
            visualization_dump = {}
        else:
            visualization_dump = None


        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
                visualization_dump=visualization_dump,
            )

            if isinstance(gaussians, dict):
                pred_depths = gaussians["depths"]
                if "depth" in batch["context"]:
                    depth_gt = batch["context"]["depth"]
                diag_maps = gaussians.get("diagnostics", None)  # eval-only per-pixel maps (Δq)
                gaussians = gaussians["gaussians"]

        # save gaussians
        if self.test_cfg.save_gaussian:
            scene = batch["scene"][0]
            if _scene_save_allowed(scene, self.test_cfg.save_scene_names):
                save_path = Path(get_cfg()['output_dir']) / 'gaussians' / (scene + '.ply')
                save_gaussian_ply(gaussians, visualization_dump, batch, save_path)

        # Ellipsoid-soup figure: draw each Gaussian as an oriented covariance ellipsoid
        # (opaque, shaded, depth-tested) from the target view, to show surface paving /
        # hole-filling. Eval-only; identical extraction for any method (means+covariances).
        if self.test_cfg.save_ellipsoids:
            from ..visualization.ellipsoid_render import render_ellipsoids, sh_dc_to_rgb
            scene = batch["scene"][0]
            if _scene_save_allowed(scene, self.test_cfg.save_scene_names):
                out_root = Path(get_cfg()["output_dir"])
                th, tw = batch["target"]["image"].shape[-2:]
                s = int(self.test_cfg.ellipsoid_scale)
                image_hw = (th * s, tw * s)
                means = gaussians.means[0]
                cov = gaussians.covariances[0]
                opac = gaussians.opacities[0]
                mode = self.test_cfg.ellipsoid_color_mode
                if mode == "scene":
                    colors = sh_dc_to_rgb(gaussians.harmonics[0])
                    render_mode = "albedo"
                elif mode == "normal":
                    colors = torch.full_like(means, 0.72)
                    render_mode = "normal"
                else:  # "flat"
                    colors = torch.full_like(means, 0.72)
                    render_mode = "albedo"
                tgt_ext = batch["target"]["extrinsics"][0]
                tgt_int = batch["target"]["intrinsics"][0]
                tgt_near = batch["target"]["near"][0]
                tgt_far = batch["target"]["far"][0]
                tgt_idx = batch["target"]["index"][0]
                glctx = None
                for vi, index in enumerate(tgt_idx):
                    for sig in self.test_cfg.ellipsoid_sigmas:
                        rgb, _ = render_ellipsoids(
                            means, cov, colors, opac,
                            tgt_ext[vi], tgt_int[vi],
                            float(tgt_near[vi]), float(tgt_far[vi]),
                            image_hw, sigma=float(sig),
                            opacity_min=self.test_cfg.ellipsoid_opacity_min,
                            bg=tuple(self.test_cfg.ellipsoid_bg),
                            color_mode=render_mode, glctx=glctx,
                        )
                        tag = ("sig%g" % sig).replace(".", "p")
                        save_image(
                            rgb.permute(2, 0, 1).contiguous(),
                            out_root / "images" / scene / "ellipsoids" / f"{int(index):0>6}_{tag}.png",
                        )

        if not self.train_cfg.forward_depth_only:
            with self.benchmarker.time("decoder", num_calls=v):

                camera_poses = batch["target"]["extrinsics"]

                if self.test_cfg.stablize_camera:
                    stable_poses = render_stabilization_path(
                        camera_poses[0].detach().cpu().numpy(),
                        k_size=self.test_cfg.stab_camera_kernel,
                    )

                    stable_poses = list(
                        map(
                            lambda x: np.concatenate(
                                (x, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0
                            ),
                            stable_poses,
                        )
                    )
                    stable_poses = torch.from_numpy(np.stack(stable_poses, axis=0)).to(
                        camera_poses
                    )
                    camera_poses = stable_poses.unsqueeze(0)

                # HRRC re-render outputs (1x is `output`). Initialized here so every render path
                # (chunked / single) leaves them defined; the single-pass branch below reassigns
                # the active ones. 2/3 = 2x/4x (standard pass); 4 = 8x (hrrc_eval_8x_only pass).
                output2 = output3 = output4 = None
                if self.test_cfg.render_chunk_size is not None:
                    chunk_size = self.test_cfg.render_chunk_size
                    num_chunks = math.ceil(camera_poses.shape[1] / chunk_size)

                    output = None
                    for i in range(num_chunks):
                        start = chunk_size * i
                        end = chunk_size * (i + 1)

                        render_intrinsics = batch["target"]["intrinsics"]
                        render_near = batch["target"]["near"]
                        render_far = batch["target"]["far"]

                        curr_output = self.decoder.forward(
                            gaussians,
                            camera_poses[:, start:end],
                            render_intrinsics[:, start:end],
                            render_near[:, start:end],
                            render_far[:, start:end],
                            (h, w),
                            depth_mode=None,
                        )

                        if i == 0:
                            output = curr_output
                        else:
                            # ignore depth
                            output.color = torch.cat(
                                (output.color, curr_output.color), dim=1
                            )

                else:
                    depth_mode = "depth"
                    output = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        depth_mode=depth_mode,
                    )
                    # 8x-only HRRC: render the SAME gaussians at 8x pixel density and skip the
                    # already-measured 2x/4x re-renders (output2/output3 stay None; their save
                    # and scoring are guarded below). output4 == 2048^2 (suffix 4 == 8x).
                    if getattr(self.test_cfg, "hrrc_eval_8x_only", False):
                        output2 = None
                        output3 = None
                        output4 = self.decoder.forward(
                            gaussians,
                            camera_poses,
                            batch["target"]["intrinsics"],
                            batch["target"]["near"],
                            batch["target"]["far"],
                            (int(h*8), int(w*8)),
                            depth_mode=depth_mode,
                        )
                    else:
                        output4 = None
                        output2 = self.decoder.forward(
                            gaussians,
                            camera_poses,
                            batch["target"]["intrinsics"],
                            batch["target"]["near"],
                            batch["target"]["far"],
                            (int(h*2), int(w*2)),
                            depth_mode=depth_mode,
                        )
                        output3 = self.decoder.forward(
                            gaussians,
                            camera_poses,
                            batch["target"]["intrinsics"],
                            batch["target"]["near"],
                            batch["target"]["far"],
                            (int(h*4), int(w*4)),
                            depth_mode=depth_mode,
                        )

        (scene,) = batch["scene"]
        self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
        path = Path(get_cfg()["output_dir"])

        # save encoder depth
        if self.test_cfg.save_depth:
            if self.train_cfg.forward_depth_only:
                depth = pred_depths[0].cpu().detach()  # [V, H, W]
            else:
                depth = (
                    visualization_dump["depth"][0, :, :, :, 0, 0].cpu().detach()
                )  # [V, H, W]

            index = batch["context"]["index"][0]

            if self.test_cfg.save_depth_concat_img:
                # concat (img0, img1, depth0, depth1)
                image = batch['context']['image'][0]  # [V, 3, H, W] in [0,1]
                image = rearrange(image, "b c h w -> h (b w) c")  # [H, VW, 3]
                image_concat = (image.detach().cpu().numpy() * 255).astype(np.uint8)  # [H, VW, 3]

                depth_concat = []

            if _scene_save_allowed(scene, self.test_cfg.save_scene_names):
                for idx, depth_i in zip(index, depth):
                    depth_viz = viz_depth_tensor(
                        1.0 / depth_i, return_numpy=True
                    )  # [H, W, 3]

                    if self.test_cfg.save_depth_concat_img:
                        depth_concat.append(depth_viz)

                    save_path = path / "images" / scene / "depth" / f"{idx:0>6}_input.png"
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    Image.fromarray(depth_viz).save(save_path)

                    # save depth as npy
                    if self.test_cfg.save_depth_npy:
                        depth_npy = depth_i.detach().cpu().numpy()
                        save_path = path / "images" / scene / "depth" / f"{idx:0>6}_input.npy"
                        save_dir = os.path.dirname(save_path)
                        os.makedirs(save_dir, exist_ok=True)
                        np.save(save_path, depth_npy)

                if self.test_cfg.save_depth_concat_img:
                    depth_concat = np.concatenate(depth_concat, axis=1)  # [H, VW, 3]
                    concat = np.concatenate((image_concat, depth_concat), axis=0)  # [2H, VW, 3]

                    save_path = path / "images" / scene / "depth" /  f"img_depth_{scene}.png"
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    Image.fromarray(concat).save(save_path)

            if "depth" in batch["target"]:
                gt_depth = batch["target"]["depth"][0]
                gt_depth2 = upsample_gt_img(gt_depth, 2.0, mode="bilinear")
                gt_depth3 = upsample_gt_img(gt_depth, 4.0, mode="bilinear")

            if self.train_cfg.forward_depth_only:
                return

        # Per-pixel Δq residual-angle map for the mechanism figure (saved with the standard
        # pass). One grayscale map per context view, normalized to a 0.5 rad (~28.6 deg) full
        # scale; the figure script color-maps and overlays it on the input view.
        if (self.test_cfg.save_depth and diag_maps is not None
                and "delta_q_angle_map" in diag_maps
                and _scene_save_allowed(scene, self.test_cfg.save_scene_names)):
            dq = diag_maps["delta_q_angle_map"][0].detach().cpu()  # [v', h, w], radians
            for idx, ang in zip(batch["context"]["index"][0], dq):
                m = (ang / 0.5).clamp(0, 1).numpy()
                out_p = path / "images" / scene / "delta_q" / f"input_{idx:0>6}.png"
                out_p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((m * 255).astype(np.uint8)).save(out_p)

        # Per-pixel in-plane anisotropy map (surfel-shape figure): s_min/s_max in [0,1], saved
        # raw as grayscale; the figure script color-maps it (low = elongated surfel).
        if (self.test_cfg.save_depth and diag_maps is not None
                and "aniso_map" in diag_maps
                and _scene_save_allowed(scene, self.test_cfg.save_scene_names)):
            am = diag_maps["aniso_map"][0].detach().cpu()  # [v', h, w]
            for idx, a_i in zip(batch["context"]["index"][0], am):
                out_p = path / "images" / scene / "aniso" / f"input_{idx:0>6}.png"
                out_p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((a_i.clamp(0, 1).numpy() * 255).astype(np.uint8)).save(out_p)

        images_prob = output.color[0]
        # 2x/4x are None on the 8x-only pass; 8x (images_prob4) is None on the standard pass.
        images_prob2 = output2.color[0] if output2 is not None else None
        images_prob3 = output3.color[0] if output3 is not None else None
        images_prob4 = output4.color[0] if output4 is not None else None
        if output.depth is not None:
            depths_prob = output.depth[0]
            depths_prob2 = output2.depth[0] if output2 is not None else None
            depths_prob3 = output3.depth[0] if output3 is not None else None
            normals_prob = output.normal[0]
            normals_prob2 = output2.normal[0] if output2 is not None else None
            normals_prob3 = output3.normal[0] if output3 is not None else None
        else:
            depths_prob = None
            depths_prob2 = None
            depths_prob3 = None
            normals_prob = None
            normals_prob2 = None
            normals_prob3 = None
            
        rgb_gt = batch["target"]["image"][0]
        # HRRC GT upsample for the test-time 2x/4x re-render metrics: bicubic by default
        # (matching SurfSplat Eq.13/Tab.1) so these numbers are directly comparable
        # to the cited bicubic-GT HRRC table. hrrc_gt_mode is one config knob shared by every
        # row (fair-comparison invariant — only the geometry-derivation method varies); clamp
        # because bicubic can overshoot [0,1] and would otherwise bias PSNR/SSIM/LPIPS. Depth
        # GT (above) stays bilinear; bicubic interpolation applies only to RGB.
        _hrrc_mode = getattr(self.train_cfg, "hrrc_gt_mode", "bicubic")
        if images_prob4 is not None:
            # 8x-only pass: upsample GT to 2048^2 (same bicubic-GT protocol, scale 8).
            rgb_gt2 = None
            rgb_gt3 = None
            rgb_gt4 = upsample_gt_img(rgb_gt, 8.0, mode=_hrrc_mode).clamp(0.0, 1.0)
        else:
            rgb_gt2 = upsample_gt_img(rgb_gt, 2.0, mode=_hrrc_mode).clamp(0.0, 1.0)
            rgb_gt3 = upsample_gt_img(rgb_gt, 4.0, mode=_hrrc_mode).clamp(0.0, 1.0)
            rgb_gt4 = None
        
        # Save images.
        if _scene_save_allowed(scene, self.test_cfg.save_scene_names):
            if self.test_cfg.save_image and images_prob4 is not None:
                # 8x-only pass: 2x/4x renders were skipped — save just 1x + 8x color.
                for index, color, color8 in zip(
                    batch["target"]["index"][0], images_prob, images_prob4
                ):
                    save_image(color, path / "images" / scene / f"color/{index:0>6}.png")
                    save_image(color8, path / "images" / scene / f"color/{index:0>6}_x8.png")
            elif self.test_cfg.save_image:
                # Chunked renders (test.render_chunk_size) produce 1x color only — depth/normal
                # and the 2x/4x re-renders stay None; substitute per-view None placeholders so
                # the save loop still runs and skips the absent resolutions.
                _n = len(images_prob)
                _seq = lambda x: x if x is not None else [None] * _n
                for index, gt, depth, depth2, depth3, normal, normal2, normal3, color, color2, color3 in zip(
                    batch["target"]["index"][0], rgb_gt, _seq(depths_prob), _seq(depths_prob2), _seq(depths_prob3),
                    _seq(normals_prob), _seq(normals_prob2), _seq(normals_prob3), images_prob, _seq(images_prob2), _seq(images_prob3)
                ):
                    save_image(color, path / "images" / scene / f"color/{index:0>6}.png")
                    if color2 is not None:
                        save_image(color2, path / "images" / scene / f"color/{index:0>6}_x2.png")
                    if color3 is not None:
                        save_image(color3, path / "images" / scene / f"color/{index:0>6}_x4.png")
                    if self.test_cfg.save_gt_image:
                        save_image(gt, path / "images" / scene / f"color/{index:0>6}_gt.png")
                    if depths_prob is not None:
                        # save_raw_depth(depth, path / "images" / scene / "depth" / f"{index:0>6}.npy")
                        # save_raw_depth(depth2, path / "images" / scene / "depth" / f"{index:0>6}_x2.npy")
                        # save_raw_depth(depth3, path / "images" / scene / "depth" / f"{index:0>6}_x4.npy")
                        depth_viz = viz_depth_tensor(1.0 / depth, return_float=True)
                        depth2_viz = viz_depth_tensor(1.0 / depth2, return_float=True)
                        depth3_viz = viz_depth_tensor(1.0 / depth3, return_float=True)
                        save_image(depth_viz, path / "images" / scene / "depth" / f"{index:0>6}.png")
                        save_image(depth2_viz, path / "images" / scene / "depth" / f"{index:0>6}_x2.png")
                        save_image(depth3_viz, path / "images" / scene / "depth" / f"{index:0>6}_x4.png")
                        normal_viz = viz_normal_tensor(normal, return_float=True)
                        normal2_viz = viz_normal_tensor(normal2, return_float=True)
                        normal3_viz = viz_normal_tensor(normal3, return_float=True)
                        save_image(normal_viz, path / "images" / scene / "normal" / f"{index:0>6}.png")
                        save_image(normal2_viz, path / "images" / scene / "normal" / f"{index:0>6}_x2.png")
                        save_image(normal3_viz, path / "images" / scene / "normal" / f"{index:0>6}_x4.png")

            # save video
            if self.test_cfg.save_video:
                frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
                save_video(
                    [a for a in images_prob],
                    path / "videos" / f"{scene}_frame_{frame_str}.mp4",
                )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v

            # 8x-only is a pure RGB re-render pass (images_prob4 set): skip depth metrics so we
            # don't create empty 2x/4x score buffers (which on_test_end would divide by zero).
            if "depth" in batch["target"] and images_prob4 is None:
                # gt_depth is otherwise only assigned inside the save_depth/save_gaussian
                # visualization block; derive here so depth metrics work for any
                # configuration, including evaluation without saved visualizations.
                gt_depth = batch["target"]["depth"][0]
                gt_depth2 = upsample_gt_img(gt_depth, 2.0, mode="bilinear")
                gt_depth3 = upsample_gt_img(gt_depth, 4.0, mode="bilinear")
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_errors(gt_depth, depths_prob)
                abs_rel2, _, _, _, a12, a22, _ = compute_depth_errors(gt_depth2, depths_prob2)
                abs_rel3, _, _, _, a13, a23, _ = compute_depth_errors(gt_depth3, depths_prob3)
                
                if f"abs_rel" not in self.test_step_outputs:
                    self.test_step_outputs[f"abs_rel"] = []
                if f"a1" not in self.test_step_outputs:
                    self.test_step_outputs[f"a1"] = []
                if f"a2" not in self.test_step_outputs:
                    self.test_step_outputs[f"a2"] = []
                if f"abs_rel2" not in self.test_step_outputs:
                    self.test_step_outputs[f"abs_rel2"] = []
                if f"a12" not in self.test_step_outputs:
                    self.test_step_outputs[f"a12"] = []
                if f"a22" not in self.test_step_outputs:
                    self.test_step_outputs[f"a22"] = []
                if f"abs_rel3" not in self.test_step_outputs:
                    self.test_step_outputs[f"abs_rel3"] = []
                if f"a13" not in self.test_step_outputs:
                    self.test_step_outputs[f"a13"] = []
                if f"a23" not in self.test_step_outputs:
                    self.test_step_outputs[f"a23"] = []         
                                   
                self.test_step_outputs[f"abs_rel"].append(abs_rel.mean().item())
                self.test_step_outputs[f"a1"].append(a1.mean().item())
                self.test_step_outputs[f"a2"].append(a2.mean().item())
                self.test_step_outputs[f"abs_rel2"].append(abs_rel2.mean().item())
                self.test_step_outputs[f"a12"].append(a12.mean().item())
                self.test_step_outputs[f"a22"].append(a22.mean().item())
                self.test_step_outputs[f"abs_rel3"].append(abs_rel3.mean().item())
                self.test_step_outputs[f"a13"].append(a13.mean().item())
                self.test_step_outputs[f"a23"].append(a23.mean().item())

                # Median-scale-aligned 1x depth diagnostic (scale-invariant): isolates
                # depth *shape* error from a global scale offset between rendered and GT
                # depth. Raw abs_rel above is the protocol metric; these are diagnostic.
                abs_rel_al, a1_al = compute_depth_errors_aligned(gt_depth, depths_prob)
                for _k in ("abs_rel_aligned", "a1_aligned"):
                    if _k not in self.test_step_outputs:
                        self.test_step_outputs[_k] = []
                self.test_step_outputs["abs_rel_aligned"].append(abs_rel_al.mean().item())
                self.test_step_outputs["a1_aligned"].append(a1_al.mean().item())

            # Normal angular error: rendered (world) normal vs GT-from-depth (camera) normal,
            # in the TARGET view. Works for any checkpoint that renders normals.
            if self.test_cfg.save_normal_npz:
                if output.normal is None:
                    raise RuntimeError(
                        "test.save_normal_npz requires decoder output.normal, but it is None"
                    )
                if "depth" not in batch["target"]:
                    raise RuntimeError(
                        "test.save_normal_npz requires batch['target']['depth'], but it is missing"
                    )

            if (
                self.test_cfg.eval_normal_error
                and not self.train_cfg.forward_depth_only
                and output.normal is not None
                and "depth" in batch["target"]
            ):
                from ..evaluation.normal_eval import (
                    gt_normal_from_depth,
                    normal_angle_errors,
                    rendered_normal_to_camera,
                )

                z_gt_t = batch["target"]["depth"][0]              # [V, H, W]
                K_t = batch["target"]["intrinsics"][0]            # [V, 3, 3]
                c2w_t = batch["target"]["extrinsics"][0]          # [V, 4, 4]
                n_gt, valid_gt = gt_normal_from_depth(
                    z_gt_t, K_t, edge_rel_thresh=self.test_cfg.normal_edge_rel_thresh
                )
                n_pred, coverage = rendered_normal_to_camera(output.normal[0], c2w_t)
                valid = (
                    valid_gt
                    & (coverage > self.test_cfg.normal_min_norm)
                    & (z_gt_t > self.test_cfg.eval_min_depth)
                    & (z_gt_t < self.test_cfg.eval_max_depth)
                )
                errs = normal_angle_errors(n_pred, n_gt, valid)
                if self.test_cfg.save_normal_npz:
                    unoriented_map = torch.full(
                        valid.shape, torch.nan, dtype=torch.float32, device=valid.device
                    )
                    oriented_map = torch.full(
                        valid.shape, torch.nan, dtype=torch.float32, device=valid.device
                    )
                    unoriented_map[valid] = errs["unoriented"].to(torch.float32)
                    oriented_map[valid] = errs["oriented"].to(torch.float32)
                    valid_gt_range = (
                        valid_gt
                        & (z_gt_t > self.test_cfg.eval_min_depth)
                        & (z_gt_t < self.test_cfg.eval_max_depth)
                    )

                    normal_npz_dir = Path(get_cfg()["output_dir"]) / "normal_npz"
                    normal_npz_dir.mkdir(parents=True, exist_ok=True)
                    normal_npz_path = normal_npz_dir / f"{scene}.npz"
                    if normal_npz_path.exists():
                        raise RuntimeError(
                            f"Refusing to overwrite existing normal dump: {normal_npz_path}"
                        )
                    np.savez_compressed(
                        normal_npz_path,
                        target_index=(
                            batch["target"]["index"][0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int32, copy=False)
                        ),
                        n_pred=n_pred.detach().to(torch.float16).cpu().numpy(),
                        n_gt=n_gt.detach().to(torch.float16).cpu().numpy(),
                        unoriented=unoriented_map.detach().cpu().numpy(),
                        oriented=oriented_map.detach().cpu().numpy(),
                        valid=valid.detach().cpu().numpy(),
                        valid_gt_range=valid_gt_range.detach().cpu().numpy(),
                    )
                    self._normal_npz_scenes.append(scene)

                if errs["unoriented"].numel() > 0:
                    self._normal_buf["oriented"].append(errs["oriented"].cpu().numpy())
                    self._normal_buf["unoriented"].append(errs["unoriented"].cpu().numpy())

                # Save camera-space GT and rendered normal maps (same frame, comparable) for the
                # ScanNet GT-normal supplement figure. Invalid GT pixels are zero -> mid-grey.
                if (self.test_cfg.save_image
                        and _scene_save_allowed(scene, self.test_cfg.save_scene_names)):
                    ng = n_gt.permute(0, 3, 1, 2)        # [V, 3, H, W] camera-space GT
                    npd = n_pred.permute(0, 3, 1, 2)     # [V, 3, H, W] camera-space rendered
                    for _i, _ng, _np in zip(batch["target"]["index"][0], ng, npd):
                        save_image(viz_normal_tensor(_ng, return_float=True),
                                   path / "images" / scene / "normal_gt" / f"{_i:0>6}.png")
                        save_image(viz_normal_tensor(_np, return_float=True),
                                   path / "images" / scene / "normal_cam" / f"{_i:0>6}.png")

            if not self.train_cfg.forward_depth_only:
                # Keyed per-scene export: record scene order so scores_*_all.json can be joined
                # back to scene hashes (figures/scripts/select_qual_scenes.py). Appended once per
                # test_step, in the same order as the score arrays below, so index i aligns.
                if self.test_cfg.compute_scores:
                    if not hasattr(self, "_scene_order"):
                        self._scene_order = []
                    self._scene_order.append(scene)
                if images_prob4 is not None:
                    # 8x-only HRRC pass. The 8x score (psnr4/ssim4/lpips4 = 2048^2) is what we came
                    # for — it depends only on the 8x render, independent of 1x. The expensive 2x/4x
                    # re-renders are skipped (already measured in the standard pass). We DO keep the
                    # cheap 1x render (256^2 ~= 1.5% of the 8x cost) and score it as psnr/ssim/lpips:
                    # a 1x PSNR that reproduces the standard pass confirms this 8x run uses the same
                    # ckpt/protocol/data, guarding the 8x column against a silently-misconfigured run.
                    for _k in ("psnr", "ssim", "lpips", "psnr4", "ssim4", "lpips4"):
                        if _k not in self.test_step_outputs:
                            self.test_step_outputs[_k] = []
                    self.test_step_outputs["psnr"].append(
                        compute_psnr(rgb_gt, images_prob).mean().item()
                    )
                    self.test_step_outputs["ssim"].append(
                        compute_ssim(rgb_gt, images_prob).mean().item()
                    )
                    self.test_step_outputs["lpips"].append(
                        compute_lpips(rgb_gt, images_prob).mean().item()
                    )
                    self.test_step_outputs["psnr4"].append(
                        compute_psnr(rgb_gt4, images_prob4).mean().item()
                    )
                    self.test_step_outputs["ssim4"].append(
                        compute_ssim(rgb_gt4, images_prob4).mean().item()
                    )
                    self.test_step_outputs["lpips4"].append(
                        compute_lpips(rgb_gt4, images_prob4).mean().item()
                    )
                    # Coverage + black-hole fraction at the two rendered scales (1x + 8x).
                    self._append_coverage(self._alpha_view(output), 1)
                    self._append_coverage(self._alpha_view(output4), 8)
                    self._append_black_frac(images_prob, rgb_gt, 1)
                    self._append_black_frac(images_prob4, rgb_gt4, 8)
                    return

                rgb = images_prob
                rgb2 = images_prob2
                rgb3 = images_prob3

                if f"psnr" not in self.test_step_outputs:
                    self.test_step_outputs[f"psnr"] = []
                if f"ssim" not in self.test_step_outputs:
                    self.test_step_outputs[f"ssim"] = []
                if f"lpips" not in self.test_step_outputs:
                    self.test_step_outputs[f"lpips"] = []
                if f"psnr2" not in self.test_step_outputs:
                    self.test_step_outputs[f"psnr2"] = []
                if f"ssim2" not in self.test_step_outputs:
                    self.test_step_outputs[f"ssim2"] = []
                if f"lpips2" not in self.test_step_outputs:
                    self.test_step_outputs[f"lpips2"] = []
                if f"psnr3" not in self.test_step_outputs:
                    self.test_step_outputs[f"psnr3"] = []
                if f"ssim3" not in self.test_step_outputs:
                    self.test_step_outputs[f"ssim3"] = []
                if f"lpips3" not in self.test_step_outputs:
                    self.test_step_outputs[f"lpips3"] = []

                self.test_step_outputs[f"psnr"].append(
                    compute_psnr(rgb_gt, rgb).mean().item()
                )
                self.test_step_outputs[f"ssim"].append(
                    compute_ssim(rgb_gt, rgb).mean().item()
                )
                self.test_step_outputs[f"lpips"].append(
                    compute_lpips(rgb_gt, rgb).mean().item()
                )
                self.test_step_outputs[f"psnr2"].append(
                    compute_psnr(rgb_gt2, rgb2).mean().item()
                )
                self.test_step_outputs[f"ssim2"].append(
                    compute_ssim(rgb_gt2, rgb2).mean().item()
                )
                self.test_step_outputs[f"lpips2"].append(
                    compute_lpips(rgb_gt2, rgb2).mean().item()
                )
                self.test_step_outputs[f"psnr3"].append(
                    compute_psnr(rgb_gt3, rgb3).mean().item()
                )
                self.test_step_outputs[f"ssim3"].append(
                    compute_ssim(rgb_gt3, rgb3).mean().item()
                )
                self.test_step_outputs[f"lpips3"].append(
                    compute_lpips(rgb_gt3, rgb3).mean().item()
                )
                # Coverage + black-hole fraction at the three scales (1x/2x/4x) on the standard pass.
                self._append_coverage(self._alpha_view(output), 1)
                self._append_coverage(self._alpha_view(output2), 2)
                self._append_coverage(self._alpha_view(output3), 4)
                self._append_black_frac(rgb, rgb_gt, 1)
                self._append_black_frac(rgb2, rgb_gt2, 2)
                self._append_black_frac(rgb3, rgb_gt3, 4)

    def on_test_end(self) -> None:
        out_dir = Path(self.test_cfg.output_path)
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            # Keyed per-scene export (see test_step): scenes_all.json[i] is the scene hash
            # for index i of every scores_*_all.json array written above.
            if getattr(self, "_scene_order", None):
                with (out_dir / "scenes_all.json").open("w") as f:
                    json.dump(self._scene_order, f)
                self._scene_order = []

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            # Reduce the rendered-normal angular errors when enabled.
            self._finalize_geometry_diagnostics(out_dir, saved_scores)

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(out_dir / "benchmark.json")
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.summarize()

        if self.test_cfg.save_normal_npz:
            self._write_normal_npz_manifest()

    def _write_normal_npz_manifest(self) -> None:
        cfg = get_cfg()
        try:
            checkpoint = cfg["checkpointing"]["load"]
            index_path = cfg["dataset"]["view_sampler"]["index_path"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "test.save_normal_npz requires checkpointing.load and "
                "dataset.view_sampler.index_path for its manifest"
            ) from exc
        if index_path is None:
            raise RuntimeError(
                "test.save_normal_npz requires dataset.view_sampler.index_path for its manifest"
            )

        index_file = Path(str(index_path))
        if not index_file.is_file():
            raise RuntimeError(f"Evaluation index does not exist: {index_file}")
        digest = hashlib.md5()
        with index_file.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)

        scene_keys = list(self._normal_npz_scenes)
        normal_npz_dir = Path(cfg["output_dir"]) / "normal_npz"
        normal_npz_dir.mkdir(parents=True, exist_ok=True)
        experiment = cfg.get("experiment", None)
        manifest = {
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "index_path": str(index_path),
            "index_md5": digest.hexdigest(),
            "num_scenes": len(scene_keys),
            "scenes": scene_keys,
            "experiment": None if experiment is None else str(experiment),
        }
        with (normal_npz_dir / "manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)

    def _finalize_geometry_diagnostics(self, out_dir: Path, saved_scores: dict) -> None:
        """Reduce normal-error buffers into scalars and an optional histogram.

        Called at evaluation end; matplotlib is imported only when saving plots.
        """
        if getattr(self.test_cfg, "eval_normal_error", False) and getattr(self, "_normal_buf", None):
            buf = self._normal_buf
            if len(buf["unoriented"]) > 0:
                un = np.concatenate(buf["unoriented"])
                ori = np.concatenate(buf["oriented"])
                saved_scores["normal_unoriented_mean_deg"] = float(np.mean(un))
                saved_scores["normal_unoriented_median_deg"] = float(np.median(un))
                saved_scores["normal_oriented_mean_deg"] = float(np.mean(ori))
                saved_scores["normal_oriented_median_deg"] = float(np.median(ori))
                saved_scores["normal_a1125"] = float((un < 11.25).mean())
                saved_scores["normal_a2250"] = float((un < 22.5).mean())
                saved_scores["normal_a30"] = float((un < 30.0).mean())
                saved_scores["normal_num_pixels"] = int(un.shape[0])

                if self.test_cfg.save_eval_plots:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(7, 4))
                    plt.hist(un, bins=45, range=(0, 90), alpha=0.8)
                    plt.axvline(11.25, color="r", ls="--"); plt.axvline(22.5, color="g", ls="--")
                    plt.xlabel("unoriented angular error (deg)"); plt.ylabel("count")
                    plt.title(f"Normal error (median={np.median(un):.2f} deg)")
                    plt.tight_layout(); plt.savefig(out_dir / "normal_histogram.png", dpi=100); plt.close()
            self._normal_buf = {k: [] for k in self._normal_buf}

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_softmax = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )

        pred_depths = None
        val_diagnostics = None

        if isinstance(gaussians_softmax, dict):
            pred_depths = gaussians_softmax["depths"]
            val_diagnostics = gaussians_softmax.get("diagnostics")
            if "depth" in batch["context"]:
                depth_gt = batch["context"]["depth"]  # [B, V, H, W]
            gaussians_softmax = gaussians_softmax["gaussians"]

        # Log residual, frame, scale, and opacity diagnostics on validation scenes.
        # This rank-zero-only hook logs locally, without a distributed reduction.
        if val_diagnostics is not None:
            for diag_name, diag_value in val_diagnostics.items():
                if torch.is_tensor(diag_value) and diag_value.dim() > 0:
                    continue  # skip non-scalar diagnostics (e.g. delta_q_angle_map, eval-only)
                self.log(f"val_diag/{diag_name}", diag_value)

        if not self.train_cfg.forward_depth_only:
            output_softmax = self.decoder.forward(
                gaussians_softmax,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
            )
            rgb_softmax = output_softmax.color[0]

            # Compute validation metrics.
            rgb_gt = batch["target"]["image"][0]
            for tag, rgb in zip(("val",), (rgb_softmax,)):
                psnr = compute_psnr(rgb_gt, rgb).mean()
                self.log(f"val/psnr_{tag}", psnr)
                lpips = compute_lpips(rgb_gt, rgb).mean()
                self.log(f"val/lpips_{tag}", lpips)
                ssim = compute_ssim(rgb_gt, rgb).mean()
                self.log(f"val/ssim_{tag}", ssim)

            # HRRC validation re-renders the same Gaussians at 2x/4x with unchanged
            # normalized intrinsics. Bicubic GT is clamped to [0,1]; the metric
            # measures image consistency. ScanNet separately measures normal error.
            if getattr(self.train_cfg, "hrrc_val", False) and batch_idx < getattr(
                self.train_cfg, "hrrc_val_scenes", 32
            ):
                hrrc_rgb, hrrc_m = {}, {}
                for scale in (2, 4):
                    out_hr = self.decoder.forward(
                        gaussians_softmax,
                        batch["target"]["extrinsics"],
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (int(h * scale), int(w * scale)),
                    )
                    rgb_hr = out_hr.color[0]
                    # bicubic GT (SurfSplat Eq.13) by default; clamp because bicubic can
                    # overshoot [0,1] and would otherwise bias PSNR/SSIM/LPIPS.
                    gt_hr = upsample_gt_img(
                        rgb_gt, float(scale),
                        mode=getattr(self.train_cfg, "hrrc_gt_mode", "bicubic"),
                    ).clamp(0.0, 1.0)
                    p = compute_psnr(gt_hr, rgb_hr).mean()
                    s = compute_ssim(gt_hr, rgb_hr).mean()
                    lp = compute_lpips(gt_hr, rgb_hr).mean()
                    hrrc_rgb[scale], hrrc_m[scale] = rgb_hr, (p, s, lp)
                    self.log(f"val/psnr_hrrc{scale}", p)
                    self.log(f"val/ssim_hrrc{scale}", s)
                    self.log(f"val/lpips_hrrc{scale}", lp)
                # Drop curves vs the 1x render (positive psnr/ssim drop or lpips rise == worse).
                self.log("val/psnr_drop_2x", psnr - hrrc_m[2][0])
                self.log("val/psnr_drop_4x", psnr - hrrc_m[4][0])
                self.log("val/ssim_drop_4x", ssim - hrrc_m[4][1])
                self.log("val/lpips_rise_4x", hrrc_m[4][2] - lpips)
                # Qualitative 1x/2x/4x panel (first scene only — avoids per-scene image spam). All
                # columns brought onto the 4x canvas so holes/discontinuities at the high render res
                # are visible and shrink as Δq activates (SurfSplat Fig.4 analogue).
                if batch_idx == 0:
                    panel = hcat(
                        add_label(vcat(*upsample_gt_img(rgb_gt, 4.0, mode="bilinear")), "GT 4x"),
                        add_label(vcat(*upsample_gt_img(rgb_softmax, 4.0, mode="bilinear")), "Pred 1x->4x"),
                        add_label(vcat(*upsample_gt_img(hrrc_rgb[2], 2.0, mode="bilinear")), "Pred 2x->4x"),
                        add_label(vcat(*hrrc_rgb[4]), "Pred 4x"),
                    )
                    self.logger.log_image(
                        "hrrc",
                        [prep_image(add_border(panel))],
                        step=self.global_step,
                        caption=batch["scene"],
                    )

        # viz depth
        if pred_depths is not None:
            # only visualize predicted depth
            pred_depths = pred_depths[0]  # [V, H, W]

            # gaussian downsample
            if pred_depths.shape[1:] != batch["context"]["image"].shape[-2:]:
                pred_depths = F.interpolate(
                    pred_depths.unsqueeze(1),
                    size=batch["context"]["image"].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(1)

            inverse_depth_pred = 1.0 / pred_depths

            concat = []
            for i in range(inverse_depth_pred.size(0)):
                concat.append(inverse_depth_pred[i])

            concat = torch.cat(concat, dim=1)  # [H, W*N]

            depth_viz = viz_depth_tensor(concat.cpu().detach())  # [3, H, W*N]

            # also concat images
            input_images = batch["context"]["image"][0]  # [N, 3, H, W]
            concat_img = [img for img in input_images]
            concat_img = torch.cat(concat_img, dim=-1) * 255  # [3, H, W*N]

            concat = torch.cat(
                (concat_img.cpu().detach(), depth_viz), dim=1
            )  # [3, H*2, W*N]

            self.logger.log_image(
                "depth",
                [concat],
                step=self.global_step,
                caption=batch["scene"],
            )

        if not self.train_cfg.forward_depth_only:
            # Construct comparison image.
            comparison = hcat(
                add_label(vcat(*batch["context"]["image"][0]), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_softmax), "Target (Prediction)"),
            )
            self.logger.log_image(
                "comparison",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )

            if not self.train_cfg.no_log_projections:
                # Render projections and construct projection image.
                projections = hcat(
                    *render_projections(
                        gaussians_softmax,
                        256,
                        extra_label="(Prediction)",
                    )[0]
                )
                self.logger.log_image(
                    "projection",
                    [prep_image(add_border(projections))],
                    step=self.global_step,
                )

                # Draw cameras.
                cameras = hcat(*render_cameras(batch, 256))
                self.logger.log_image(
                    "cameras", [prep_image(add_border(cameras))], step=self.global_step
                )

            if self.encoder_visualizer is not None:
                for k, image in self.encoder_visualizer.visualize(
                    batch["context"], self.global_step
                ).items():
                    self.logger.log_image(k, [prep_image(image)], step=self.global_step)

            # Run video validation step.
            if not self.train_cfg.no_viz_video:
                self.render_video_interpolation(batch)
                self.render_video_wobble(batch)
                if self.train_cfg.extended_visualization:
                    self.render_video_interpolation_exaggerated(batch)

    def on_validation_epoch_end(self) -> None:
        """Evaluate the complete validation set at the configured interval."""
        if self.trainer.sanity_checking and self.global_rank == 0:
            print(self.encoder)  # log the model to wandb log files

        if (not self.trainer.sanity_checking) and (self.eval_data_cfg is not None):
            self.eval_cnt = self.eval_cnt + 1
            if self.eval_cnt % self.train_cfg.eval_model_every_n_val == 0:
                # backup current ckpt before running full test sets eval
                if self.train_cfg.eval_save_model:
                    ckpt_saved_path = (
                        self.trainer.checkpoint_callback.format_checkpoint_name(
                            dict(
                                epoch=self.trainer.current_epoch,
                                step=self.trainer.global_step,
                            )
                        )
                    )
                    backup_dir = str(
                        Path(ckpt_saved_path).parent.parent / "checkpoints_backups"
                    )
                    if self.global_rank == 0:
                        os.makedirs(backup_dir, exist_ok=True)
                    ckpt_saved_path = os.path.join(
                        backup_dir, os.path.basename(ckpt_saved_path)
                    )
                    # call save_checkpoint on ALL process as suggested by pytorch_lightning
                    self.trainer.save_checkpoint(
                        ckpt_saved_path,
                        weights_only=True,
                    )
                    if self.global_rank == 0:
                        print(f"backup model to {ckpt_saved_path}.")

                # run full test sets eval on rank=0 device
                self.run_full_test_sets_eval()

    @rank_zero_only
    def run_full_test_sets_eval(self) -> None:
        start_t = time.time()

        pred_depths = None
        depth_gt = None

        full_testsets = self.trainer.datamodule.test_dataloader(
            dataset_cfg=self.eval_data_cfg
        )
        scores_dict = {}

        if not self.train_cfg.forward_depth_only:
            for score_tag in ("psnr", "ssim", "lpips"):
                scores_dict[score_tag] = {}
                for method_tag in ("deterministic", "probabilistic"):
                    scores_dict[score_tag][method_tag] = []

        # evaluate depth
        if self.train_cfg.viz_depth:
            for score_tag in ("abs_rel", "rmse", "a1"):
                scores_dict[score_tag] = {}
                for method_tag in ("deterministic", "probabilistic"):
                    scores_dict[score_tag][method_tag] = []

        self.benchmarker.clear_history()
        time_skip_first_n_steps = min(
            self.train_cfg.eval_time_skip_steps, len(full_testsets)
        )
        time_skip_steps_dict = {"encoder": 0, "decoder": 0}
        for batch_idx, batch in tqdm(
            enumerate(full_testsets),
            total=min(len(full_testsets), self.train_cfg.eval_data_length),
        ):
            if batch_idx >= self.train_cfg.eval_data_length:
                break

            batch = self.data_shim(batch)
            batch = self.transfer_batch_to_device(batch, "cuda", dataloader_idx=0)

            # Render Gaussians.
            b, v, _, h, w = batch["target"]["image"].shape
            assert b == 1
            if batch_idx < time_skip_first_n_steps:
                time_skip_steps_dict["encoder"] += 1
                time_skip_steps_dict["decoder"] += v

            with self.benchmarker.time("encoder"):
                gaussians_probabilistic = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=False,
                )

                if isinstance(gaussians_probabilistic, dict):
                    pred_depths = gaussians_probabilistic["depths"]
                    if "depth" in batch["context"]:
                        depth_gt = batch["context"]["depth"]
                    gaussians_probabilistic = gaussians_probabilistic["gaussians"]

            if not self.train_cfg.forward_depth_only:
                with self.benchmarker.time("decoder", num_calls=v):
                    output_probabilistic = self.decoder.forward(
                        gaussians_probabilistic,
                        batch["target"]["extrinsics"],
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                    )
                rgbs = [output_probabilistic.color[0]]
                tags = ["probabilistic"]

                if self.train_cfg.eval_deterministic:
                    gaussians_deterministic = self.encoder(
                        batch["context"],
                        self.global_step,
                        deterministic=True,
                    )
                    output_deterministic = self.decoder.forward(
                        gaussians_deterministic,
                        batch["target"]["extrinsics"],
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                    )
                    rgbs.append(output_deterministic.color[0])
                    tags.append("deterministic")

                # Compute validation metrics.
                rgb_gt = batch["target"]["image"][0]
                for tag, rgb in zip(tags, rgbs):
                    scores_dict["psnr"][tag].append(
                        compute_psnr(rgb_gt, rgb).mean().item()
                    )
                    scores_dict["lpips"][tag].append(
                        compute_lpips(rgb_gt, rgb).mean().item()
                    )
                    scores_dict["ssim"][tag].append(
                        compute_ssim(rgb_gt, rgb).mean().item()
                    )

        # summarise scores and log to logger
        for score_tag, methods in scores_dict.items():
            for method_tag, cur_scores in methods.items():
                if len(cur_scores) > 0:
                    cur_mean = sum(cur_scores) / len(cur_scores)
                    self.log(f"test/{score_tag}", cur_mean)
        # summarise run time
        for tag, times in self.benchmarker.execution_times.items():
            times = times[int(time_skip_steps_dict[tag]) :]
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
            self.log(f"test/runtime_avg_{tag}", np.mean(times))
        self.benchmarker.clear_history()

        overall_eval_time = time.time() - start_t
        print(f"Eval total time cost: {overall_eval_time:.3f}s")
        self.log("test/runtime_all", overall_eval_time)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        if isinstance(gaussians_prob, dict):
            gaussians_prob = gaussians_prob["gaussians"]

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]

        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Prediction"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        if self.optimizer_cfg.lr_monodepth > 0:
            pretrained_params = []
            new_params = []

            for name, param in self.named_parameters():
                if "pretrained" in name:
                    pretrained_params.append(param)
                else:
                    new_params.append(param)

            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": pretrained_params,
                        "lr": self.optimizer_cfg.lr_monodepth,
                    },
                    {"params": new_params, "lr": self.optimizer_cfg.lr},
                ],
                weight_decay=self.optimizer_cfg.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                [self.optimizer_cfg.lr_monodepth, self.optimizer_cfg.lr],
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        else:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.optimizer_cfg.lr,
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
