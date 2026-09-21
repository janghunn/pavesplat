#!/usr/bin/env python3
"""Generate the ScanNet zero-shot evaluation index (supplement section 3).

Uses SurfSplat frame slicing and the repository overlap geometry, seed 0.
The frozen protocol has 635 slices, with GT depth available for 633.

The shipped assets/evaluation_index_scannet.json is the protocol of record.
This generator reproduces it verbatim from the raw poses. Its numeric frame
ordering differs from convert_scan.py's lexicographic chunk ordering for
non-zero-padded frame names. A regenerated index must be paired with chunks
produced by that converter, as the shipped index was. The index/chunk pairing
is fixed by the shipped files.
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from src.geometry.projection import get_world_rays, sample_image_grid
from src.geometry.epipolar_lines import project_rays

# --- ScanNet conversion constants (mirror convert_scan.py exactly) ---
INTR_W, INTR_H = 640, 480     # intrinsic_depth.txt normalization (depth-sensor res)
SLICE = 350                   # nominal frames per slice (short tails are merged)
MERGE_TAIL = 100              # if n-end < this, merge tail into current slice
OVERLAP = 20                  # slice overlap


def load_scene_cameras(scene_dir: Path):
    """Return cameras[N,18] for finite poses in numeric frame order."""
    m = np.loadtxt(scene_dir / "intrinsic" / "intrinsic_depth.txt", dtype=np.float32)
    intr = np.array([m[0, 0] / INTR_W, m[1, 1] / INTR_H,
                     m[0, 2] / INTR_W, m[1, 2] / INTR_H, 0.0, 0.0], dtype=np.float32)
    pose_files = sorted((scene_dir / "pose").glob("*.txt"), key=lambda p: int(p.stem))
    cams = []
    for pf in pose_files:
        c2w = np.loadtxt(pf, dtype=np.float32)
        if not np.isfinite(c2w).all():
            continue
        w2c = np.linalg.inv(c2w)[:3].flatten().tolist()
        cams.append(np.concatenate([intr, w2c]))
    if not cams:
        return np.zeros((0, 18), dtype=np.float32)
    return np.stack(cams).astype(np.float32)


def slices_of(n: int):
    """Yield (slice_idx, start, end) exactly as convert_scan.py."""
    start, sidx = 0, 0
    while start < n:
        end = min(start + SLICE, n)
        if n - end < MERGE_TAIL and end != n:
            end = n
        yield sidx, start, end
        if end != n:
            start = end - OVERLAP
            sidx += 1
        else:
            start = end


def convert_poses(poses: torch.Tensor):
    """cameras[b,18] -> (extrinsics c2w [b,4,4], intrinsics [b,3,3]); mirrors dataset_scan."""
    b = poses.shape[0]
    intr = torch.eye(3, dtype=torch.float32).repeat(b, 1, 1)
    fx, fy, cx, cy = poses[:, :4].T
    intr[:, 0, 0], intr[:, 1, 1], intr[:, 0, 2], intr[:, 1, 2] = fx, fy, cx, cy
    w2c = torch.eye(4, dtype=torch.float32).repeat(b, 1, 1)
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intr


def find_pair_and_targets(extrinsics, intrinsics, h, w, device, cfg, generator):
    """Select context views by mutual projected-ray overlap, then sample targets."""
    v = extrinsics.shape[0]
    if not 1 <= cfg.num_target_views <= v:
        raise ValueError(
            f"Cannot sample {cfg.num_target_views} distinct targets from a {v}-frame slice; "
            "num_target_views must be positive and no greater than the frame count."
        )
    max_target_capacity = 0
    xy, _ = sample_image_grid((h, w), device)
    xy = rearrange(xy, "h w xy -> (h w) xy")
    for context_index in torch.randperm(v, generator=generator):
        c_origins, c_dirs = get_world_rays(xy, extrinsics[context_index], intrinsics[context_index])
        valid = []
        for step in (1, -1):
            cur = context_index + step * cfg.min_distance
            while 0 <= cur.item() < v:
                cur_origins, cur_dirs = get_world_rays(xy, extrinsics[cur], intrinsics[cur])
                onto_cur = project_rays(c_origins, c_dirs, extrinsics[cur], intrinsics[cur])
                onto_ctx = project_rays(cur_origins, cur_dirs, extrinsics[context_index], intrinsics[context_index])
                oa = onto_ctx["overlaps_image"].float().mean()
                ob = onto_cur["overlaps_image"].float().mean()
                overlap = min(oa, ob)
                delta = (cur - context_index).abs()
                if cfg.min_overlap <= overlap <= cfg.max_overlap:
                    capacity = delta.item() + 1
                    max_target_capacity = max(max_target_capacity, capacity)
                    if capacity >= cfg.num_target_views:
                        valid.append(cur.item())
                if overlap < cfg.min_overlap or delta > cfg.max_distance:
                    break
                cur += step
        if valid:
            chosen = valid[torch.randint(0, len(valid), tuple(), generator=generator)]
            left, right = min(chosen, context_index.item()), max(chosen, context_index.item())
            while True:
                tv = torch.randint(left, right + 1, (cfg.num_target_views,), generator=generator)
                if (tv.unique(return_counts=True)[1] == 1).all():
                    break
            return (left, right), tuple(sorted(tv.tolist()))
    if max_target_capacity:
        raise ValueError(
            f"No context pair in this {v}-frame slice can provide "
            f"{cfg.num_target_views} distinct targets; the largest eligible interval "
            f"contains {max_target_capacity} frames."
        )
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_test", required=True, help="scannet_raw/test dir")
    ap.add_argument("--out", required=True, help="output index json path")
    ap.add_argument("--image_hw", type=int, default=256, help="grid resolution for overlap (eval res)")
    ap.add_argument("--num_target_views", type=int, default=3)
    ap.add_argument("--min_distance", type=int, default=45)
    ap.add_argument("--max_distance", type=int, default=192)
    ap.add_argument("--min_overlap", type=float, default=0.6)
    ap.add_argument("--max_overlap", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    cfg = ap.parse_args()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    raw = Path(cfg.raw_test)
    scenes = sorted([d.name for d in raw.iterdir() if d.is_dir()])
    h = w = cfg.image_hw

    index, n_none = {}, 0
    for scene in tqdm(scenes, desc="scenes"):
        cams = load_scene_cameras(raw / scene)
        n = cams.shape[0]
        if n == 0:
            continue
        for sidx, start, end in slices_of(n):
            key = f"{scene}_slice{sidx:02d}"
            sl = torch.from_numpy(cams[start:end])
            extr, intr = convert_poses(sl)
            extr, intr = extr.to(device), intr.to(device)
            res = find_pair_and_targets(extr, intr, h, w, device, cfg, gen)
            if res is None:
                index[key] = None
                n_none += 1
            else:
                ctx, tgt = res
                index[key] = {"context": list(ctx), "target": list(tgt)}

    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(index, f)
    valid = sum(1 for v in index.values() if v is not None)
    print(f"wrote {cfg.out}: {len(index)} slices, {valid} valid, {n_none} None")


if __name__ == "__main__":
    main()
