from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F

from ....geometry.projection import get_world_rays, unproject
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance, quaternion_to_matrix


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def _percentile(x, q: float):
    """No-grad scalar percentile for diagnostics. Flattens to fp32 and subsamples huge
    tensors (torch.quantile rejects > 2**24 elements) with a deterministic even stride.
    Returns a 0-dim tensor (NaN on an empty input)."""
    x = x.detach().reshape(-1).float()
    n = x.numel()
    if n == 0:
        return x.new_tensor(float("nan"))
    cap = 1 << 20
    if n > cap:
        idx = torch.linspace(0, n - 1, cap, device=x.device).long()
        x = x[idx]
    return torch.quantile(x, q)


def pts2covariance(points, tangent_frame: bool = False, eps: float = 1e-8):
    """Build surface frames from a point map [b, v, h, w, 3] (Eqs. 2–4).

    The default uses SurfSplat's normal-only Rodrigues frame. With tangent_frame,
    R_a = [e1, e2, n] is orthonormal and the in-plane radii follow tangent magnitudes.
    Returns rotations [b,v,h,w,3,3], radii [b,v,h,w,2], normals [b,v,h,w,3].
    """
    b, v, h, w, c = points.shape
    assert c == 3, "3-d coord."

    sobel_x = torch.tensor([[-1/8, 0, 1/8], [-2/8, 0, 2/8], [-1/8, 0, 1/8]], dtype=torch.float32, device=points.device).view(1, 1, 3, 3).repeat(3,1,1,1)
    sobel_y = torch.tensor([[-1/8, -2/8, -1/8], [0, 0, 0], [1/8, 2/8, 1/8]], dtype=torch.float32, device=points.device).view(1, 1, 3, 3).repeat(3,1,1,1)

    # [n * v, c, h, w]
    points = points.view(b * v, h, w, c).permute(0, 3, 1, 2)  # [(b v), c, h, w]
    depths = points[:, 2, :, :]  # [(b v), h, w]

    grad_x = F.conv2d(points, sobel_x, groups=c)
    grad_y = F.conv2d(points, sobel_y, groups=c)

    if tangent_frame:
        # Restore full resolution before constructing the tangent-aligned frame.
        t_u = F.pad(grad_x, (1, 1, 1, 1), mode="replicate")  # ∂X/∂x  (image-x surface tangent)
        t_v = F.pad(grad_y, (1, 1, 1, 1), mode="replicate")  # ∂X/∂y  (image-y surface tangent)
        return _tangent_frame_covariance(t_u, t_v, b, v, h, w, eps)

    normals = torch.cross(grad_x, grad_y, dim=1)
    normals = F.pad(normals, (1, 1, 1, 1), mode='replicate')  
    normals = normals.permute(0, 2, 3, 1).view(b, v, h, w, c)  
    normals = F.normalize(normals, dim=-1)

    rotations = build_gs_rotation_from_normal(normals.reshape(b*v*h*w, c))
    rotations = rotations.view(b, v, h, w, c, c)
    
    grad_x = F.pad(grad_x, (1, 1, 1, 1), mode='replicate').clamp(min=-0.1,max=0.1)
    grad_y = F.pad(grad_y, (1, 1, 1, 1), mode='replicate').clamp(min=-0.1,max=0.1)
    
    delta_x_x = grad_x[:, 0, :, :].reshape(-1) # (b v h w)
    delta_x_y = grad_y[:, 0, :, :].reshape(-1)
    delta_y_x = grad_x[:, 1, :, :].reshape(-1)
    delta_y_y = grad_y[:, 1, :, :].reshape(-1)
    delta_z_x = grad_x[:, 2, :, :].reshape(-1)
    delta_z_y = grad_y[:, 2, :, :].reshape(-1)

    scale_x = torch.sqrt(delta_x_x**2 + delta_z_x**2)
    scale_y = torch.sqrt(delta_y_y**2 + delta_z_y**2)  
    scales = torch.stack((scale_x, scale_y), dim=-1)  # [batch, h, w, 2]
    scales = scales.view(b, v, h, w, 2)  # [batch, v, h, w, 2]

    return rotations, scales, normals


def _tangent_frame_covariance(t_u, t_v, b, v, h, w, eps: float):
    """Tangent-aligned frame R_a = [e1, e2, n] and radii (Supp. Alg. 1).

    e1 follows the image-x tangent projected onto the surface; e2 = n × e1.
    The camera-facing normal follows the tangent cross product. Colinear tangents
    use the optical axis, and a tangent parallel to the normal uses a reference axis.
    Returns rotations [b,v,h,w,3,3], radii [b,v,h,w,2], normals [b,v,h,w,3].
    """
    n_pix = b * v * h * w
    t_u = t_u.permute(0, 2, 3, 1).reshape(n_pix, 3)
    t_v = t_v.permute(0, 2, 3, 1).reshape(n_pix, 3)

    # Surface normal from the tangent cross product.
    cross = torch.cross(t_u, t_v, dim=-1)
    cross_norm = cross.norm(dim=-1, keepdim=True)
    n = cross / cross_norm.clamp_min(eps)
    # Degeneracy guard: when the tangents are (near-)colinear (1-D ridge/crease, textureless,
    # depth edge) the cross product collapses to ~0 and a plain normalise would return a NON-unit
    # / zero normal, collapsing the frame (det -> 0, rank-deficient covariance) and breaking the
    # rasterizer's PSD contract. Fall back to the optical axis +z, which is exactly what the
    # SurfSplat baseline does for a zero normal (build_gs_rotation_from_normal(0) = I, 3rd column
    # +z) — so a degenerate pixel degrades gracefully instead of corrupting the Gaussian.
    z_axis = n.new_tensor([0.0, 0.0, 1.0]).expand_as(n)
    n = torch.where(cross_norm < 1e-6, z_axis, n)
    # camera-facing +z convention. sign(0) -> +1 keeps n unit at nz == 0.
    sign = torch.where(n[:, 2:3] < 0, -1.0, 1.0)
    n = n * sign

    # In-plane axis e1 = tangent projected off the normal. Where the tangent is (near-)
    # parallel to n (grazing / textureless), fall back to a reference axis so the frame stays
    # a valid orthonormal rotation; e1's magnitude there is irrelevant (scale -> ~0 -> clamp).
    e1_raw = t_u - (t_u * n).sum(-1, keepdim=True) * n
    e1_len = e1_raw.norm(dim=-1, keepdim=True)
    degenerate = e1_len < 1e-6
    # reference axis least aligned with n (x-axis, or y-axis when n ~ x)
    near_x = n[:, 0:1].abs() > 0.9
    ref = torch.where(
        near_x,
        n.new_tensor([0.0, 1.0, 0.0]).expand_as(n),
        n.new_tensor([1.0, 0.0, 0.0]).expand_as(n),
    )
    e1_fallback = F.normalize(ref - (ref * n).sum(-1, keepdim=True) * n, dim=-1, eps=eps)
    e1 = torch.where(degenerate, e1_fallback, e1_raw / e1_len.clamp_min(eps))
    e1 = F.normalize(e1, dim=-1, eps=eps)
    e2 = F.normalize(torch.cross(n, e1, dim=-1), dim=-1, eps=eps)
    e1 = torch.cross(e2, n, dim=-1)  # re-orthogonalise (already unit; e2 × n = e1)

    rotations = torch.stack([e1, e2, n], dim=-1)  # columns -> [n_pix, 3, 3]

    # Tangent-magnitude in-plane radii on the e1 / e2 axes. The tangents are clamped per
    # component (matching the baseline's ±0.1 guard) so depth edges cannot explode the scale;
    # the final gaussian_scale clamp downstream still applies.
    t_u_c = t_u.clamp(-0.1, 0.1)
    t_v_c = t_v.clamp(-0.1, 0.1)
    s1 = t_u_c.norm(dim=-1)
    t_v_perp = t_v_c - (t_v_c * e1).sum(-1, keepdim=True) * e1
    s2 = t_v_perp.norm(dim=-1)

    rotations = rotations.view(b, v, h, w, 3, 3)
    scales = torch.stack([s1, s2], dim=-1).view(b, v, h, w, 2)
    normals = n.view(b, v, h, w, 3)
    return rotations, scales, normals


def build_gs_rotation_from_normal(normal):
    """
    input normalized normal
    default normal is (0,0,1)
    """
    start_normal = torch.tensor([0., 0., 1.], device=normal.device, dtype=normal.dtype).repeat(normal.shape[0], 1)
    normal = normal * torch.sign(normal[:, 2:3])
    # Compute cross product and dot product
    v = torch.cross(start_normal, normal, dim=1)
    c = torch.sum(start_normal * normal, dim=1)
    s = torch.norm(v, dim=1)

    vx = torch.zeros((start_normal.shape[0], 3, 3), device=start_normal.device, dtype=start_normal.dtype)
    vx[:, 0, 1] = -v[:, 2]
    vx[:, 0, 2] = v[:, 1]
    vx[:, 1, 0] = v[:, 2]
    vx[:, 1, 2] = -v[:, 0]
    vx[:, 2, 0] = -v[:, 1]
    vx[:, 2, 1] = v[:, 0]

    # Compute the rotation matrix using Rodrigues' rotation formula
    I = torch.eye(3, device=start_normal.device, dtype=start_normal.dtype).unsqueeze(0).repeat(start_normal.shape[0], 1, 1)
    R = I + vx * s.unsqueeze(1).unsqueeze(2) + torch.bmm(vx, vx) * ((1 - c).unsqueeze(1).unsqueeze(2))
    return R


def build_quaternion_from_rotation(rot, eps: float = 1e-8):
    """Rotation matrices [N, 3, 3] -> quaternions in the repo's w-last [x, y, z, w]
    convention (matches gaussians.py::quaternion_to_matrix and scipy R.from_quat in
    ply_export).

    Shepperd-style max-pivot conversion. The naive ``w = sqrt(1+trace)/2, xyz/(4w)``
    form is NaN at trace ~ -1 (180-degree rotations — reachable for the tangent-aligned
    frame, e.g. e1 ~ -x with n ~ +z at a depth edge, and for the Δq-composed final
    frame) and catastrophically inaccurate near it. Picking the largest of the four
    pivots keeps the denominator >= 1 in the selected branch, so the result is finite
    and accurate for every proper rotation. Output is sign-normalised to w >= 0 (the
    quaternion double cover; matches the old convention) and unit-normalised (the
    SurfSplat analytic frame is only an approximate rotation, so its raw conversion
    need not be unit). Consumers are export/visualization only (ply, point-cloud
    figures) — the render path builds covariances directly from the matrices."""
    m00, m01, m02 = rot[:, 0, 0], rot[:, 0, 1], rot[:, 0, 2]
    m10, m11, m12 = rot[:, 1, 0], rot[:, 1, 1], rot[:, 1, 2]
    m20, m21, m22 = rot[:, 2, 0], rot[:, 2, 1], rot[:, 2, 2]

    # 2|w|, 2|x|, 2|y|, 2|z| — clamp_min(0): float error can push 1+trace slightly negative.
    q_abs = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    ).clamp_min(0.0).sqrt()

    # Candidate (w, x, y, z) numerators per pivot; row k is exact where q_abs[:, k] is
    # the max (its denominator 2*q_abs[:, k] is then >= 1 for a proper rotation).
    quat_by_pivot = torch.stack(
        [
            torch.stack([q_abs[:, 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[:, 1] ** 2, m01 + m10, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m01 + m10, q_abs[:, 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[:, 3] ** 2], dim=-1),
        ],
        dim=-2,
    )  # [N, 4 pivots, 4 comps (w-first)]
    quat_candidates = quat_by_pivot / (2.0 * q_abs[:, :, None].clamp_min(eps))

    best = q_abs.argmax(dim=-1)
    wxyz = quat_candidates[torch.arange(rot.shape[0], device=rot.device), best]
    wxyz = F.normalize(wxyz, dim=-1, eps=eps)
    xyzw = wxyz[:, [1, 2, 3, 0]]  # -> w-last
    return torch.where(xyzw[:, 3:4] < 0, -xyzw, xyzw)

def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None):
    """
    Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.
    
    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    with torch.no_grad():
        shape = depth.shape
        depth = depth.detach().reshape(-1, 1, *shape[-2:])
        if mask is not None:
            mask = mask.reshape(-1, 1, *shape[-2:])

        if mask is None:
            diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
        else:
            diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

        edge = torch.zeros_like(depth, dtype=torch.bool)
        if atol is not None:
            edge |= diff > atol
        if rtol is not None:
            edge |= (diff / depth).nan_to_num_() > rtol
        edge = edge.reshape(*shape)
        return edge

@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    scales_mul: Float[Tensor, "*batch 2"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]
    normals: Float[Tensor, "*batch 3"]
    # Optional side-channel for training diagnostics (e.g. mean |Δq angle|).
    # Default None keeps the baseline path and the parent adapter unaffected.
    diagnostics: dict | None = None

@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int
    scl_mul_range: float = 3.0
    # Coordinate-consistent camera-space construction; False preserves SurfSplat.
    camera_space_frame: bool = False
    # Tangent-aligned frame R_a = [e1, e2, n] with tangent-magnitude radii.
    tangent_frame: bool = False
    # Identity-initialized residual reorientation q = q_a ⊗ Δq (Eq. 5).
    use_rotation_residual: bool = False


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"] | None,
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"] | None,
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
    ) -> Gaussians:
        scales, rotations, sh = raw_gaussians.split((2, 4, 3 * self.d_sh), dim=-1)
        scales_mul = torch.ones_like(scales)
        scales_z = self.cfg.gaussian_scale_min * torch.ones_like(scales[..., :1])
        scales = torch.cat((scales, scales_z), dim=-1)
        scales = torch.clamp(F.softplus(scales - 4.),
            min=self.cfg.gaussian_scale_min,
            max=self.cfg.gaussian_scale_max,
            )

        assert input_images is not None

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # [2, 2, 65536, 1, 1, 3, 25]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        if input_images is not None:
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        opacities_max = 0.6
        opacities = opacities * opacities_max

        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            opacities=opacities,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
            scales_mul=scales_mul,
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 6 + 3 * self.d_sh



class SurfelAdapter(GaussianAdapter):
    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"] | None,
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"] | None,
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
    ) -> Gaussians:

        if self.cfg.use_rotation_residual:
            # Four extra channels carry the rotation residual Δq (w-last [x,y,z,w]).
            scales_mul, delta_q, sh = raw_gaussians.split(
                (2, 4, 3 * self.d_sh), dim=-1
            )
        else:
            scales_mul, sh = raw_gaussians.split((2, 3 * self.d_sh), dim=-1)
        scales_mul_range = self.cfg.scl_mul_range
        scales_mul = torch.clamp(F.softplus(scales_mul),
            min=1/scales_mul_range,
            max=scales_mul_range,
        )

        assert input_images is not None
        b,v,c,h,w = input_images.shape
        # Normalize the quaternion features to yield a valid quaternion.
        # rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # [2, 2, 65536, 1, 1, 3, 25]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        if input_images is not None:
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)

        # Compute Gaussian means (world space — these are the Gaussian centres).
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        # Choose the point cloud from which the local surface frame is derived.
        if self.cfg.camera_space_frame:
            # Coordinate-consistent formulation: build the frame in camera space so the
            # camera-facing flip (sign of z == optical axis) is meaningful, and let c2w be
            # applied exactly once below.
            cam_dirs = unproject(
                coordinates, torch.ones_like(coordinates[..., 0]), intrinsics
            )
            cam_dirs = cam_dirs / cam_dirs[..., -1:]
            frame_pts = cam_dirs * depths[..., None]
        else:
            # Baseline (SurfSplat repro): frame built in WORLD space, then c2w applied
            # again below — the original double-rotation behaviour, preserved on purpose.
            frame_pts = means

        frame_pts = rearrange(
            frame_pts, "b v (h w) srf spp c -> b (v srf spp) h w c", h=h, w=w
        )
        means = rearrange(means, "b v (h w) srf spp c -> b (v srf spp) h w c", h=h, w=w)
        rotations, scales, normals = pts2covariance(
            frame_pts, tangent_frame=self.cfg.tangent_frame, eps=eps,
        )

        scales_mul = rearrange(scales_mul, "b v (h w) srf spp c -> b (v srf spp) h w c", h=h, w=w)
        scales = scales * scales_mul

        # In-plane radius diagnostics before the scale clamp (Eq. 4).
        with torch.no_grad():
            scale_xy_mean = scales.mean()
            scale_xy_max = scales.max()
            scale_clamp_min_frac = (scales <= self.cfg.gaussian_scale_min).float().mean()
            scale_clamp_max_frac = (scales >= self.cfg.gaussian_scale_max).float().mean()
            # In-plane anisotropy: s_min/s_max near one indicates a round surfel.
            _aniso = scales.amin(dim=-1) / scales.amax(dim=-1).clamp_min(eps)
            scale_aniso_ratio = _aniso.mean()
            isotropic_frac = (_aniso > 0.9).float().mean()

        scales_z = self.cfg.gaussian_scale_min * torch.ones_like(scales[..., :1])
        scales = torch.cat((scales, scales_z), dim=-1)
        
        scales = torch.clamp(scales,
        min=self.cfg.gaussian_scale_min,
        max=self.cfg.gaussian_scale_max,
        )

        # Residual reorientation: R = R_a @ R_delta (Eq. 5); identity preserves R_a.
        diagnostics = None
        if self.cfg.use_rotation_residual:
            delta_q = rearrange(
                delta_q, "b v (h w) srf spp c -> b (v srf spp) h w c", h=h, w=w
            )
            delta_q = delta_q / (delta_q.norm(dim=-1, keepdim=True) + eps)  # w-last [x,y,z,w]
            r_delta = quaternion_to_matrix(delta_q)                          # [b, v, h, w, 3, 3]
            rotations = rotations @ r_delta
            # The reoriented frame's third column gives the surface normal.
            # Normalize because the SurfSplat frame is only approximately orthonormal.
            normals = F.normalize(rotations[..., :, 2], dim=-1)

            # Residual diagnostic: the residual rotation angle |Δq|.
            # angle = 2·acos(|w|) of the unit quaternion. Detached — logging only.
            with torch.no_grad():
                qw = delta_q[..., 3].abs().clamp(max=1.0)
                delta_q_angle = 2.0 * torch.acos(qw)
                diagnostics = {
                    "delta_q_angle": delta_q_angle.mean(),
                    "delta_q_angle_max": delta_q_angle.max(),
                    # Tail of the residual-angle distribution: the mean can stay ~0 while a few
                    # pixels carry the correction. p90/p99 separate "Δq collapsed" from
                    # "Δq active only at hard pixels".
                    "delta_q_angle_p90": _percentile(delta_q_angle, 0.90),
                    "delta_q_angle_p99": _percentile(delta_q_angle, 0.99),
                    # Fraction of pixels with a non-trivial residual (> ~0.057°). Stays well above 0
                    # while Δq is active; -> 0 alongside a ~0 mean indicates a collapsed residual.
                    "delta_q_nonzero_frac": (delta_q_angle > 1e-3).float().mean(),
                }
                # Per-pixel residual-angle map for the mechanism figure (eval only — never
                # added during training, so training-time scalar logging is unchanged). The
                # model_wrapper log loops skip non-scalar diagnostics; test_step saves it.
                if not self.training:
                    diagnostics["delta_q_angle_map"] = delta_q_angle  # [b, v', h, w], radians

        covariances = (rotations
            @ scales.diag_embed()
            @ rearrange(scales.diag_embed(), "... i j -> ... j i")
            @ rearrange(rotations, "... i j -> ... j i"))
        # Export/visualization-only product (ply, point-cloud figures): nothing backprops
        # through quats (the loss-side types.Gaussians drops the field), so keep the
        # 4-pivot conversion's [N, 4, 4] intermediates out of the autograd graph.
        with torch.no_grad():
            quats = build_quaternion_from_rotation(rotations.view(b * v * h * w, c, c))
        quats = quats.view(b, v, h, w, 4)
        
        means = rearrange(means, "b v h w c -> b v (h w) () () c")
        quats = rearrange(quats, "b v h w c -> b v (h w) () () c")
        scales = rearrange(scales, "b v h w c -> b v (h w) () () c")
        covariances = rearrange(covariances, "b v h w c1 c2 -> b v (h w) () () c1 c2")
        scales_mul = rearrange(scales_mul, "b v h w c -> b v (h w) () () c")
        normals = rearrange(normals, "b v h w c -> b v (h w) () () c")
                
        # Create world-space covariance matrices.
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        if self.cfg.camera_space_frame:
            # The frame (and its surface normal) were built in camera space; rotate the
            # stored normal into world space so the rendered-normal fake-colour keeps the
            # baseline's world-space convention.
            normals = (c2w_rotations @ normals[..., None]).squeeze(-1)

        opacities_max = 0.6
        raw_opacities = opacities
        opacities = opacities * opacities_max

        # Frame, scale, and opacity diagnostics; detached from the rendering path.
        with torch.no_grad():
            if diagnostics is None:
                diagnostics = {}
            lo, hi = 1.0 / scales_mul_range, scales_mul_range
            diagnostics["scales_mul_mean"] = scales_mul.mean()
            diagnostics["scales_mul_min_frac"] = (scales_mul <= lo * 1.001).float().mean()
            diagnostics["scales_mul_max_frac"] = (scales_mul >= hi * 0.999).float().mean()
            diagnostics["opacity_mean"] = opacities.mean()              # post-0.6-cap opacity
            diagnostics["opacity_sat_frac"] = (raw_opacities > 0.99).float().mean()  # cap-saturated
            R = rotations.reshape(-1, 3, 3)
            eye = torch.eye(3, device=R.device, dtype=R.dtype)
            diagnostics["frame_orth_err"] = (R.transpose(-1, -2) @ R - eye).abs().mean()
            frame_det = torch.linalg.det(R)
            diagnostics["frame_det_mean"] = frame_det.mean()
            diagnostics["frame_det_min"] = frame_det.min()        # < 0 flags a handedness flip
            diagnostics["scale_xy_mean"] = scale_xy_mean
            diagnostics["scale_xy_max"] = scale_xy_max
            diagnostics["scale_clamp_min_frac"] = scale_clamp_min_frac
            diagnostics["scale_clamp_max_frac"] = scale_clamp_max_frac
            diagnostics["scale_aniso_ratio"] = scale_aniso_ratio
            diagnostics["isotropic_frac"] = isotropic_frac
            # Per-pixel in-plane anisotropy map for the surfel-shape figure (eval only; the log
            # loops skip non-scalar diagnostics). Low ratio = elongated surfel; high = round.
            if not self.training:
                diagnostics["aniso_map"] = _aniso  # [b, v, h, w], s_min/s_max in [0,1]

        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            opacities=opacities,
            scales=scales,
            rotations=quats,
            scales_mul=scales_mul,
            normals=normals,
            diagnostics=diagnostics,
        )
        
    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 2 + (4 if self.cfg.use_rotation_residual else 0) + 3 * self.d_sh