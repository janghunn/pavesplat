"""Ellipsoid-soup renderer: draw each 3D Gaussian as an oriented covariance ellipsoid.

Motivation (PaveSplat figure): the normal alpha-blended splat render hides *how* the
surfels are laid out. Rendering each Gaussian as an opaque, depth-tested, shaded
ellipsoid surface makes the surface-paving directly visible -- you can see whether the
oriented surfels tile the surface or leave background-coloured holes between them.

The ellipsoid for a Gaussian with covariance Sigma is the iso-surface
    {x : (x-mu)^T Sigma^{-1} (x-mu) = sigma^2}
i.e. a unit sphere mapped by  V * diag(sigma * sqrt(lambda))  where Sigma = V diag(lambda) V^T.
We build it directly from the covariance, so the same code is method-agnostic
(DepthSplat / SurfSplat / PaveSplat all expose `means` + `covariances`).

This module is intentionally self-contained (no project-internal imports) so it can be
copied verbatim into the DepthSplat baseline repo for the third figure column.

Rasterization uses nvdiffrast's headless CUDA context (no display / EGL needed).
Install:  pip install nvdiffrast ninja
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

SH_C0 = 0.28209479177387814  # 0-th order real SH; matches gaussian_adapter.RGB2SH


# ---------------------------------------------------------------------------
# Unit icosphere (subdivision of an icosahedron) -- no trimesh dependency.
# subdivisions=1 -> 42 verts / 80 faces; smooth enough for small ellipsoids.
# ---------------------------------------------------------------------------
def _unit_icosphere(subdivisions: int = 1, device="cuda", dtype=torch.float32):
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = torch.tensor(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=dtype, device=device,
    )
    faces = torch.tensor(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=torch.long, device=device,
    )

    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}
        new_faces = []
        vlist = [verts]
        n = [verts.shape[0]]

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key in cache:
                return cache[key]
            m = 0.5 * (vlist[0][a] + vlist[0][b])
            vlist[0] = torch.cat([vlist[0], m[None]], dim=0)
            idx = n[0]
            n[0] += 1
            cache[key] = idx
            return idx

        for f in faces.tolist():
            a, b, c = f
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = vlist[0]
        faces = torch.tensor(new_faces, dtype=torch.long, device=device)

    verts = verts / verts.norm(dim=-1, keepdim=True)  # project onto unit sphere
    return verts, faces


# ---------------------------------------------------------------------------
# Camera: build an OpenGL clip-space projection that matches the splat renderer.
# The splat path (cuda_splatting.render_cuda) uses get_fov() + a *symmetric* frustum
# (principal point assumed centred), so we mirror that to guarantee identical framing.
# ---------------------------------------------------------------------------
def fov_from_intrinsics(intrinsics: Tensor) -> tuple[float, float]:
    """Normalized intrinsics (fx,fy as fraction of W,H) -> (fov_x, fov_y) radians."""
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    fov_x = 2.0 * math.atan(0.5 / fx)
    fov_y = 2.0 * math.atan(0.5 / fy)
    return fov_x, fov_y


def _gl_projection(fov_x: float, fov_y: float, near: float, far: float, device, dtype):
    """Standard OpenGL symmetric perspective (NDC z in [-1,1], camera looks down -z)."""
    p = torch.zeros((4, 4), device=device, dtype=dtype)
    p[0, 0] = 1.0 / math.tan(0.5 * fov_x)
    p[1, 1] = 1.0 / math.tan(0.5 * fov_y)
    p[2, 2] = -(far + near) / (far - near)
    p[2, 3] = -2.0 * far * near / (far - near)
    p[3, 2] = -1.0
    return p


# OpenCV camera (x right, y down, z forward) -> OpenGL camera (x right, y up, z back).
_CV2GL = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))


@torch.no_grad()
def render_ellipsoids(
    means: Tensor,            # [G, 3] world
    covariances: Tensor,      # [G, 3, 3] world, SPD
    colors: Tensor,           # [G, 3] linear RGB in [0,1]
    opacities: Tensor,        # [G]
    extrinsics_c2w: Tensor,   # [4, 4] camera-to-world
    intrinsics_norm: Tensor,  # [3, 3] normalized
    near: float,
    far: float,
    image_hw: tuple[int, int],
    sigma: float = 2.0,
    opacity_min: float = 0.05,
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ambient: float = 0.35,
    subdivisions: int = 1,
    color_mode: str = "albedo",   # "albedo" (colors + shading) | "normal" (world-normal RGB)
    glctx=None,
) -> tuple[Tensor, Tensor]:
    """Render Gaussians as opaque shaded ellipsoids from one camera.

    Returns (rgb[H,W,3] in [0,1], alpha[H,W] in {0,1}) on the input device.
    """
    import nvdiffrast.torch as dr

    device = means.device
    dtype = torch.float32
    H, W = image_hw
    # nvdiffrast CUDA raster wants multiples of 8.
    Hp, Wp = (H + 7) // 8 * 8, (W + 7) // 8 * 8

    keep = opacities >= opacity_min
    means, covariances, colors = means[keep], covariances[keep], colors[keep]
    G = means.shape[0]
    if G == 0:
        rgb = torch.tensor(bg, device=device, dtype=dtype).expand(H, W, 3).contiguous()
        return rgb, torch.zeros(H, W, device=device, dtype=dtype)

    # Eigendecomposition: Sigma = V diag(lambda) V^T  (eigh -> ascending, V orthonormal).
    lam, V = torch.linalg.eigh(covariances.to(torch.float64))
    lam = lam.clamp_min(1e-12)
    axis = (sigma * lam.sqrt()).to(dtype)         # [G, 3] semi-axis lengths
    V = V.to(dtype)                                # [G, 3, 3] columns = eigvecs

    base_v, base_f = _unit_icosphere(subdivisions, device=device, dtype=dtype)  # [P,3],[F,3]
    P = base_v.shape[0]

    # Per-ellipsoid linear map M = V * diag(axis); vertices = mean + M @ base_v.
    M = V * axis[:, None, :]                                   # [G,3,3] (scale columns)
    verts = torch.einsum("gij,pj->gpi", M, base_v) + means[:, None, :]  # [G,P,3]
    # Normals transform by inverse-transpose of M = V * diag(1/axis) applied to base_v.
    Minvt = V * (1.0 / axis)[:, None, :]                      # [G,3,3]
    normals = torch.einsum("gij,pj->gpi", Minvt, base_v)      # [G,P,3] world normals
    normals = F.normalize(normals, dim=-1)

    verts = verts.reshape(G * P, 3)
    normals = normals.reshape(G * P, 3)
    vcol = colors[:, None, :].expand(G, P, 3).reshape(G * P, 3)

    faces = base_f[None] + (torch.arange(G, device=device) * P)[:, None, None]  # [G,F,3]
    faces = faces.reshape(-1, 3).to(torch.int32)

    # World -> OpenGL clip.
    fov_x, fov_y = fov_from_intrinsics(intrinsics_norm)
    proj = _gl_projection(fov_x, fov_y, near, far, device, dtype)
    view_gl = _CV2GL.to(device, dtype) @ torch.linalg.inv(extrinsics_c2w.to(dtype))
    mvp = proj @ view_gl                                       # [4,4]
    verts_h = torch.cat([verts, torch.ones(verts.shape[0], 1, device=device, dtype=dtype)], -1)
    clip = (mvp @ verts_h.T).T.contiguous()                   # [N,4]

    if glctx is None:
        glctx = dr.RasterizeCudaContext(device=device)
    rast, _ = dr.rasterize(glctx, clip[None], faces, resolution=[Hp, Wp])  # [1,Hp,Wp,4]

    mask = (rast[..., 3] > 0).float()                         # [1,Hp,Wp]
    # Interpolate per-vertex colour and world normal across the triangles.
    col_px, _ = dr.interpolate(vcol[None], rast, faces)       # [1,Hp,Wp,3]
    nrm_px, _ = dr.interpolate(normals[None], rast, faces)    # [1,Hp,Wp,3]
    nrm_px = F.normalize(nrm_px, dim=-1)

    # Headlight Lambertian: brighter where the surface faces the camera.
    cam_dir_world = extrinsics_c2w[:3, 2].to(dtype)           # OpenCV forward (+z) in world
    cam_dir_world = F.normalize(cam_dir_world, dim=0)
    ndl = (nrm_px * cam_dir_world).sum(-1).abs().clamp(0, 1)  # [1,Hp,Wp]
    shade = (ambient + (1.0 - ambient) * ndl)[..., None]      # [1,Hp,Wp,1]

    bg_t = torch.tensor(bg, device=device, dtype=dtype)
    if color_mode == "normal":
        img = 0.5 * (nrm_px + 1.0)            # world-normal RGB, no shading
    else:
        img = col_px * shade
    img = img * mask[..., None] + bg_t * (1.0 - mask[..., None])
    img = dr.antialias(img, rast, clip[None], faces)          # smooth silhouettes

    # nvdiffrast emits rows bottom-first (OpenGL); flip to a top-left-origin image.
    rgb = torch.flip(img, dims=[1])[0].clamp(0, 1)            # [Hp,Wp,3]
    alpha = torch.flip(mask, dims=[1])[0]                     # [Hp,Wp]
    if (Hp, Wp) != (H, W):  # rounded-up raster res -> resize back to the requested size
        rgb = F.interpolate(rgb.permute(2, 0, 1)[None], size=(H, W),
                            mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        alpha = F.interpolate(alpha[None, None], size=(H, W),
                              mode="bilinear", align_corners=False)[0, 0]
    return rgb.contiguous(), alpha.contiguous()


def sh_dc_to_rgb(harmonics: Tensor) -> Tensor:
    """harmonics [..., 3, d_sh] -> RGB [..., 3] in [0,1] from the DC term only."""
    return (SH_C0 * harmonics[..., 0] + 0.5).clamp(0, 1)


# ---------------------------------------------------------------------------
# Self-test: render a synthetic plane of ellipsoids to verify camera convention.
#   python -m src.visualization.ellipsoid_render /tmp/ellipsoid_selftest.png
# Expect: a slab of shaded ellipsoids filling the centre, the +x ellipsoid (red) on the
# RIGHT and the +y ellipsoid (green) at the BOTTOM (OpenCV y-down image convention).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import numpy as np
    from PIL import Image

    dev = "cuda"
    # Grid of flat disks on the z=2 plane (in front of an identity camera at origin).
    xs = torch.linspace(-1, 1, 11, device=dev)
    ys = torch.linspace(-1, 1, 11, device=dev)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    means = torch.stack([gx.flatten(), gy.flatten(), torch.full_like(gx.flatten(), 2.0)], -1)
    G = means.shape[0]
    base = torch.diag(torch.tensor([0.12, 0.12, 0.01], device=dev)) ** 2  # thin disks
    cov = base[None].expand(G, 3, 3).contiguous()
    col = torch.full((G, 3), 0.7, device=dev)
    # Mark axes: +x red, +y green.
    col[(means[:, 0] > 0.8) & (means[:, 1].abs() < 0.15)] = torch.tensor([1.0, 0, 0], device=dev)
    col[(means[:, 1] > 0.8) & (means[:, 0].abs() < 0.15)] = torch.tensor([0, 1.0, 0], device=dev)
    opac = torch.ones(G, device=dev)
    c2w = torch.eye(4, device=dev)
    K = torch.tensor([[0.7, 0, 0.5], [0, 0.7, 0.5], [0, 0, 1.0]], device=dev)

    rgb, _ = render_ellipsoids(
        means, cov, col, opac, c2w, K, near=0.1, far=100.0,
        image_hw=(512, 512), sigma=2.0, opacity_min=0.0, bg=(0.05, 0.05, 0.08),
    )
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ellipsoid_selftest.png"
    Image.fromarray((rgb.cpu().numpy() * 255).astype(np.uint8)).save(out)
    print(f"wrote {out}  ({G} ellipsoids)")
