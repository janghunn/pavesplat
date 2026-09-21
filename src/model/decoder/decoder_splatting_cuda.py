from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda, render_normal_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda", "splatting_cuda_test"]
    # SurfSplat issue #9: separate train/eval alpha thresholds for the premultiplied-alpha
    # normalization in render_cuda. A larger floor while training prevents the 2D Gaussians
    # from collapsing (too small / too transparent -> black holes); the small value is used
    # at inference to preserve render quality. Identical for every row -> fair comparison.
    alpha_threshold_train: float = 0.1
    alpha_threshold_eval: float = 0.001


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        # Coverage diagnostic is eval-only: request the accumulated-alpha map only when
        # not training, so the training path stays byte-identical (no extra stack, no
        # extra tensor held in the autograd graph).
        return_alpha = not self.training
        rendered = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            alpha_threshold=(
                self.cfg.alpha_threshold_train
                if self.training
                else self.cfg.alpha_threshold_eval
            ),
            return_alpha=return_alpha,
        )
        if return_alpha:
            color, alpha = rendered
            # alpha is [(b v) (1) h w]; fold (b v) -> b v, then drop a singleton channel if the
            # rasterizer returns [1, h, w] per view, yielding [b, v, h, w] (== DecoderOutput.alpha).
            alpha = rearrange(alpha, "(b v) ... -> b v ...", b=b, v=v)
            if alpha.dim() == 5:
                alpha = alpha[:, :, 0]
        else:
            color = rendered
            alpha = None
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        return DecoderOutput(
            color,
            None
            if depth_mode is None
            else self.render_depth(
                gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode
            ),
            None
            if depth_mode is None
            else self.render_normal(
                gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode
            ),
            alpha=alpha,
        )

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)
    
        
    def render_normal(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view 3 height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_normal_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            repeat(gaussians.normals, "b g xyz -> (b v) g xyz", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) c h w -> b v c h w", b=b, v=v)

