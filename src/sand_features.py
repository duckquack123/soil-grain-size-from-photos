"""Features for the dedicated SAND model at a finer common scale (H16).

Why: test sands are loose clean sands (H12); our 24 training soils never look like that. SandSnap
(McFall et al. 2024, Zenodo 7063226, CC-BY-4.0) has 141 sieved beach sands photographed with
smartphones (19-46 px/mm, ~30 mm field of view) that look like the test sands. Its field of view is
too small for the 73 mm / 4.6 px/mm pipeline, so this script works at SAND_PPM = 12 px/mm with
224 px (18.7 mm) tiles. Both SandSnap and the test photos are sharper than 12 px/mm, so nothing is
upsampled. Competition training photos (4.6 px/mm) are NOT used here.

Per image: DINOv2-base (CLS + mean patch, grayscale-normalised like --gray) and texture band
energies at 0.1-4 mm, averaged over tiles.
usage: python src/sand_features.py   ->  cache/sand_feats.npz
"""
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import AutoModel

from common import CACHE_DIR, effective_ppm, list_images, test_name_to_id

SAND_PPM = 12.0
TILE = 224
SIGMAS_MM = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
SANDSNAP = "/scratch/m25csa007/external/sandsnap"
MAX_TILES = 16
dev = "cuda" if torch.cuda.is_available() else "cpu"
VER = os.environ.get("SAND_VER", "v2")  # v1: 75 SandSnap samples (site zips only); v2: + tiles set1 (141)


def load(path, ppm):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = SAND_PPM / ppm
    if s > 1.02:
        raise ValueError(f"{path}: {ppm:.1f} px/mm is coarser than {SAND_PPM} — would need upsampling")
    if s < 0.98:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BOX)
    return np.asarray(im)


def tiles(img):
    H, W = img.shape[:2]
    pos = [(y, x) for y in range(0, H - TILE + 1, TILE // 2) for x in range(0, W - TILE + 1, TILE // 2)]
    if len(pos) > MAX_TILES:
        pos = [pos[i] for i in np.linspace(0, len(pos) - 1, MAX_TILES).astype(int)]
    return np.stack([img[y:y + TILE, x:x + TILE] for y, x in pos]) if pos else None


def texture(tile):
    g = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = g / (cv2.GaussianBlur(g, (0, 0), TILE / 4) + 1e-3)
    bl = [g] + [cv2.GaussianBlur(g, (0, 0), s * SAND_PPM) for s in SIGMAS_MM]
    en = np.array([np.abs(bl[i] - bl[i + 1]).mean() for i in range(len(SIGMAS_MM))])
    return np.concatenate([np.log(en + 1e-5), en / en.sum()])


@torch.no_grad()
def main():
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    idx = pd.read_csv(f"{SANDSNAP}/sandsnap_index.csv")
    items = [("sandsnap", f"{SANDSNAP}/{r.local}", 1 / r.mm_px, r.sample_key, int(r.dry)) for r in idx.itertuples()]
    _, te = list_images()
    items += [("test", f, effective_ppm(f), test_name_to_id(f), -1) for f in te]
    rec = []
    for src, path, ppm, key, dry in items:
        try:
            T = tiles(load(path, ppm))
        except ValueError as e:
            print("skip", e); continue
        if T is None:
            print("skip (too small)", path); continue
        x = torch.from_numpy(T).to(dev).permute(0, 3, 1, 2).float() / 255
        g = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
        g = (g - g.mean((2, 3), keepdim=True)) / (g.std((2, 3), keepdim=True) + 1e-4)
        x = (((0.45 + 0.2 * g).clamp(0, 1).expand(-1, 3, -1, -1) - mean) / std).half()
        h = model(pixel_values=x).last_hidden_state
        emb = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().mean(0).cpu().numpy()
        tex = np.mean([texture(t) for t in T], 0)
        rec.append(dict(src=src, path=path, key=key, dry=dry, ppm=ppm, ntiles=len(T), emb=emb, tex=tex))
    print({s: sum(r["src"] == s for r in rec) for s in ("sandsnap", "test")})
    np.savez(f"{CACHE_DIR}/sand_feats_{VER}.npz", **{k: np.array([r[k] for r in rec]) for k in rec[0]})
    print(f"saved cache/sand_feats_{VER}.npz")


if __name__ == "__main__":
    main()
