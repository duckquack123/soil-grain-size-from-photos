"""Segment visible grains with SAM and measure them in mm (images at TARGET_PPM px/mm).

Per image we store, for a ladder of sieve sizes, the fraction of the soil area covered by
grains whose sieve size (minor side of the min-area rectangle) exceeds that size. By
Delesse's principle the area fraction of a section estimates the volume (≈ mass) fraction,
so these are physically interpretable estimates of "% coarser than d" for the visible part.

SAM resizes its input to 1024 px, so besides the full image we also segment a 2x2 grid of
overlapping windows (≈1.7x finer) to catch small grains; window masks touching an interior
window edge are dropped so truncated grains do not bias sizes downwards.

usage: python sam_grains.py OUT.npz [--shard i --nshards n] [--only substr ...]
"""
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from common import CACHE_DIR, FIG_DIR, camera_of, in_dir, list_images, sample_id_train, test_name_to_id
from features import TARGET_PPM, load_scaled

SIZES_MM = np.array([2, 3, 4, 6.3, 10, 14, 20, 31.5, 45, 63, 90, 125])
SAM_KW = dict(points_per_batch=64, pred_iou_thresh=0.86, stability_score_thresh=0.9, crops_n_layers=0)


def soil_mask(img):
    v = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 2]
    k = np.ones((int(4 * TARGET_PPM) | 1,) * 2, np.uint8)
    bad = cv2.morphologyEx(((v < 50) | (v > 245)).astype(np.uint8), cv2.MORPH_OPEN, k)
    bad = cv2.dilate(bad, k)
    return bad == 0


def as_grain(m, y0, x0, soil):
    """Crop a full-window mask to its bbox and test it is a compact grain inside the soil.

    Returns (y, x, crop_mask, sieve_mm, eqdiam_mm) in image coordinates, or None.
    """
    ys, xs = np.nonzero(m)
    if len(ys) < 30:
        return None
    ya, yb, xa, xb = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = m[ya:yb, xa:xb]
    y, x = y0 + ya, x0 + xa
    a = crop.sum()
    if a > 0.25 * soil.sum() or (crop & soil[y:y + crop.shape[0], x:x + crop.shape[1]]).sum() < 0.9 * a:
        return None
    cnts, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    if hull <= 0 or a / hull < 0.8:  # grains are compact; matrix / shadow / merged regions are not
        return None
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    return y, x, crop, min(w, h) / TARGET_PPM, 2 * np.sqrt(a / np.pi) / TARGET_PPM


def segment(gen, img, soil):
    H, W = soil.shape
    cands = []
    for m in gen(Image.fromarray(img), **SAM_KW)["masks"]:
        g = as_grain(np.asarray(m, bool), 0, 0, soil)
        if g:
            cands.append(g)
    wy, wx = int(H * 0.6), int(W * 0.6)
    for y0 in (0, H - wy):
        for x0 in (0, W - wx):
            for m in gen(Image.fromarray(img[y0:y0 + wy, x0:x0 + wx]), **SAM_KW)["masks"]:
                m = np.asarray(m, bool)
                # drop masks cut by an interior window edge (edges on the image border are fine)
                if ((y0 > 0 and m[0].any()) or (y0 + wy < H and m[-1].any()) or
                        (x0 > 0 and m[:, 0].any()) or (x0 + wx < W and m[:, -1].any())):
                    continue
                g = as_grain(m, y0, x0, soil)
                if g:
                    cands.append(g)
    # dedupe: larger grains first, skip candidates mostly covered by already-kept grains
    cands.sort(key=lambda g: -g[2].sum())
    taken = np.zeros((H, W), bool)
    keep = []
    for g in cands:
        y, x, crop = g[:3]
        sl = taken[y:y + crop.shape[0], x:x + crop.shape[1]]
        if (crop & sl).sum() > 0.2 * crop.sum():
            continue
        sl |= crop
        keep.append(g)
    return keep


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()
    gen = pipeline("mask-generation", model="facebook/sam-vit-large", device=0)  # fp16 breaks the NMS step
    tr, te = list_images()
    files = [("train", f) for f in tr] + [("test", f) for f in te]
    files = [sf for i, sf in enumerate(files) if i % args.nshards == args.shard]
    if args.only:
        files = [(s, f) for s, f in files if any(o in f for o in args.only)]
    rec = []
    for split, f in files:
        img = load_scaled(f)
        soil = soil_mask(img)
        grains = segment(gen, img, soil)
        g = np.array([(s, e, c.sum()) for _, _, c, s, e in grains]).reshape(-1, 3)
        A = soil.sum()
        frac = np.array([g[g[:, 0] > s, 2].sum() / A for s in SIZES_MM])
        frac_eq = np.array([g[g[:, 1] > s, 2].sum() / A for s in SIZES_MM])
        sid = sample_id_train(f) if split == "train" else test_name_to_id(f)
        rec.append(dict(path=f, split=split, sample=sid, cam=camera_of(f), frac=frac, frac_eq=frac_eq,
                        ngrains=len(g), grain_area=g[:, 2].sum() / A,
                        grains=g, soil_px=A))  # per-grain (sieve_mm, eqdiam_mm, area_px) for re-weighting (H9)
        print(f"{split} {sid:28s} {camera_of(f):24s} n={len(g):4d} cover={g[:, 2].sum() / A:.2f} "
              f">4mm={frac[2]:.2f} >20mm={frac[6]:.2f} >63mm={frac[9]:.2f}", flush=True)
        if args.only:  # overlay for visual inspection
            ov = img.copy()
            rng = np.random.default_rng(0)
            for y, x, c, _, _ in grains:
                sl = ov[y:y + c.shape[0], x:x + c.shape[1]]
                sl[c] = (0.5 * sl[c] + 0.5 * rng.integers(0, 255, 3)).astype(np.uint8)
            Image.fromarray(ov).save(f"{FIG_DIR}/debug/sam_{sid}_{camera_of(f)}.jpg", quality=85)
    arr = {k: np.array([r[k] for r in rec]) for k in rec[0] if k != "grains"}
    arr["grains"] = np.empty(len(rec), dtype=object)
    arr["grains"][:] = [r["grains"] for r in rec]
    np.savez(in_dir(CACHE_DIR, args.out), **arr)
    print("saved", args.out)


if __name__ == "__main__":
    main()
