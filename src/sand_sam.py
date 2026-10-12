"""H15 (attempt 2): segment individual sand grains with SAM on native-resolution crops.

Attempt 1 (morphological granulometry, sand_granulometry.py) failed sanity checks: same distribution
for every sample incl. cobble surfaces, and Kleinkummerfeld 2-2 (visibly coarser) came out finer than
18-3. Here each grain is an explicit mask that can be checked visually.

Small native crops (CROP_MM) are upscaled to SAM's 1024 px input so 0.15 mm grains are ~15 px; a dense
point grid (POINTS_PER_SIDE) gives every grain >= ~0.15 mm a prompt. Grain masks must be compact
(solidity >= 0.85), not touch the crop border (truncation bias) and be < 2 mm wide (coarser grains
come from the 4.6 px/mm SAM pipeline). Sieve size = shorter side of the min-area rectangle.

Per photo we store all grain (sieve_mm, area_mm2) pairs and the crop area, so any weighting can be
recomputed later. usage: python src/sand_sam.py OUT.npz [--shard i --nshards n] [--only substr ...]
"""
import argparse

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import pipeline

from common import CACHE_DIR, FIG_DIR, camera_of, effective_ppm, in_dir, list_images, test_name_to_id
from sand_granulometry import crop_positions

CROP_MM = 10.0
N_CROPS = 8
POINTS_PER_SIDE = 64
SAM_KW = dict(points_per_batch=64, pred_iou_thresh=0.86, stability_score_thresh=0.9, crops_n_layers=0,
              points_per_side=POINTS_PER_SIDE)


def grains_in_crop(gen, crop, ppm):
    """Return list of (sieve_mm, area_mm2, mask) for compact, untruncated grains < 2 mm in a native crop."""
    h, w = crop.shape[:2]
    up = Image.fromarray(crop).resize((1024, int(1024 * h / w)), Image.BICUBIC)
    scale = up.width / w                                                   # SAM px per native px
    out = []
    for m in gen(up, **SAM_KW)["masks"]:
        m = np.asarray(m, bool)
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            continue
        a = m.sum()
        if a < 20 or a > 0.25 * m.size:
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull <= 0 or a / hull < 0.85:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        sieve = min(rw, rh) / scale / ppm
        if sieve >= 2.0:
            continue
        out.append((sieve, a / scale ** 2 / ppm ** 2, m))
    # dedupe: larger first, skip masks mostly covered by kept ones (SAM returns parts and wholes)
    out.sort(key=lambda g: -g[1])
    taken = np.zeros_like(out[0][2]) if out else None
    keep = []
    for s, a, m in out:
        if (m & taken).sum() > 0.2 * m.sum():
            continue
        taken |= m
        keep.append((s, a, m))
    return keep, up


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()
    gen = pipeline("mask-generation", model="facebook/sam-vit-large", device=0)
    _, te = list_images()
    files = [f for i, f in enumerate(te) if i % args.nshards == args.shard]
    if args.only:
        files = [f for f in files if any(o in f for o in args.only)]
    rec = []
    for f in files:
        ppm = effective_ppm(f)
        img = np.asarray(ImageOps.exif_transpose(Image.open(f)).convert("RGB"))
        c = int(CROP_MM * ppm)
        pos = crop_positions(f, img.shape[:2], ppm)[:: max(1, 12 // N_CROPS)][:N_CROPS]
        sieves, areas, crop_area = [], [], 0.0
        for k, (y, x) in enumerate(pos):
            crop = img[y:y + c, x:x + c]
            g, up = grains_in_crop(gen, crop, ppm)
            sieves += [s for s, _, _ in g]
            areas += [a for _, a, _ in g]
            crop_area += CROP_MM ** 2
            if args.only and k < 2:  # overlay for visual inspection
                ov = np.asarray(up).copy()
                rng = np.random.default_rng(k)
                for _, _, m in g:
                    ov[m] = (0.45 * ov[m] + 0.55 * rng.integers(0, 255, 3)).astype(np.uint8)
                sid = test_name_to_id(f)
                Image.fromarray(np.hstack([np.asarray(up), ov])).save(
                    f"{FIG_DIR}/debug/sand_sam_{sid}_{k}.jpg", quality=88)
        s, a = np.array(sieves), np.array(areas)
        sid = test_name_to_id(f)
        cov = a.sum() / crop_area if crop_area else 0
        d50 = np.interp(0.5, np.cumsum(a[np.argsort(s)]) / a.sum(), np.sort(s)) if len(s) else np.nan
        rec.append(dict(path=f, sample=sid, ppm=ppm, sieve=s, area=a, crop_area=crop_area))
        print(f"{sid:28s} {camera_of(f):9s} crops={len(pos)} grains={len(s):5d} coverage={cov:.2f} "
              f"area-D50={d50:.2f}mm  frac(<0.2)={a[s < 0.2].sum() / max(a.sum(), 1e-9):.2f} "
              f"frac(<0.63)={a[s < 0.63].sum() / max(a.sum(), 1e-9):.2f}", flush=True)
    np.savez(in_dir(CACHE_DIR, args.out), **{k: np.array([r[k] for r in rec], dtype=object) for k in rec[0]})
    print("saved", args.out)


if __name__ == "__main__":
    main()
