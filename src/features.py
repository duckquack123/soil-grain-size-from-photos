"""Extract per-image features on a common physical scale.

Every image is resampled to TARGET_PPM px/mm, cut into tiles of TILE_MM, tiles touching the
tray / lamp are dropped, and per tile we compute
  * DINOv2 CLS + mean-patch embedding (GPU)
  * scale-space texture energies in mm units (illumination-normalised, camera-agnostic)
Results are saved per image; aggregation to samples happens in the model script.

usage: python features.py [MODEL] [OUT.npz] [--gray] [--degrade]
  --gray     colour-invariant: DINOv2 sees per-tile contrast-normalised grayscale, no saturation feature
             (training photos are warm-lit, test photos neutral grey)
  --degrade  equalise image quality: blur + JPEG re-encode every scaled image (training photos are
             app-compressed, test photos are pristine full-resolution iPhone files)
"""
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoModel

from common import (CACHE_DIR, TEST_DIR, TRAIN_DIR, camera_of, effective_ppm, in_dir, list_images,
                    sample_id_train, test_name_to_id)

TARGET_PPM = 4.6
TILE_PX = 336
TILE_MM = TILE_PX / TARGET_PPM
STRIDE = TILE_PX // 2
MAX_TILES = 40
SIGMAS_MM = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
_pos = [a for a in sys.argv[1:] if not a.startswith("--")] if __name__ == "__main__" else []
FLAGS = {a for a in sys.argv[1:] if a.startswith("--")} if __name__ == "__main__" else set()
MODEL = _pos[0] if len(_pos) > 0 else "facebook/dinov2-base"
OUT = _pos[1] if len(_pos) > 1 else "feats_dinov2base.npz"
GRAY, DEGRADE = "--gray" in FLAGS, "--degrade" in FLAGS
dev = "cuda" if torch.cuda.is_available() else "cpu"


def degrade(img):
    """Blur slightly and JPEG re-encode so all images share the training photos' quality."""
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def load_scaled(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = TARGET_PPM / effective_ppm(path)
    if abs(s - 1) > 0.02:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS if s > 1 else Image.BOX)
    return np.asarray(im)


def valid_tiles(img):
    """Top-left corners of tiles free of tray / lamp (very dark or blown-out pixels)."""
    v = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 2]
    # only large dark / saturated regions (tray, lamp) count; small inter-grain shadows and
    # specular highlights are removed by the opening (kernel ~4 mm)
    k = np.ones((int(4 * TARGET_PPM) | 1,) * 2, np.uint8)
    bad = cv2.morphologyEx(((v < 50) | (v > 245)).astype(np.uint8), cv2.MORPH_OPEN, k).astype(np.float32)
    ii = cv2.integral(bad)
    H, W = bad.shape
    out = []
    for y in range(0, H - TILE_PX + 1, STRIDE // 2):
        for x in range(0, W - TILE_PX + 1, STRIDE // 2):
            f = (ii[y + TILE_PX, x + TILE_PX] - ii[y, x + TILE_PX] - ii[y + TILE_PX, x] + ii[y, x]) / TILE_PX ** 2
            if f < 0.02:
                out.append((y, x))
    if len(out) > MAX_TILES:
        out = [out[i] for i in np.linspace(0, len(out) - 1, MAX_TILES).astype(int)]
    return out


def hand_features(tile):
    g = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = g / (cv2.GaussianBlur(g, (0, 0), TILE_PX / 6) + 1e-3)  # remove illumination gradient
    blurs = [g] + [cv2.GaussianBlur(g, (0, 0), s * TARGET_PPM) for s in SIGMAS_MM]
    en = np.array([np.abs(blurs[i] - blurs[i + 1]).mean() for i in range(len(SIGMAS_MM))])  # DoG band energy
    var = np.array([(blurs[i] - blurs[i + 1]).std() for i in range(len(SIGMAS_MM))])
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx ** 2 + gy ** 2)
    gq = np.quantile(gm, [0.5, 0.75, 0.9, 0.97])
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV).astype(np.float32)
    col = np.array([hsv[..., 1].mean() / 255, hsv[..., 2].std() / (hsv[..., 2].mean() + 1e-3)])
    if GRAY:  # saturation is lighting / white-balance dependent
        col = col[1:]
    p = en / (en.sum() + 1e-6)
    return np.concatenate([np.log(en + 1e-4), np.log(var + 1e-4), p, np.log(gq + 1e-4), col, [g.std()]])


@torch.no_grad()
def main():
    proc_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    proc_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval().half()
    tr, te = list_images()
    rec = []
    for split, files in (("train", tr), ("test", te)):
        for f in files:
            img = load_scaled(f)
            if DEGRADE:
                img = degrade(img)
            tiles = valid_tiles(img)
            if not tiles:  # fall back to the central crop
                H, W = img.shape[:2]
                tiles = [((H - TILE_PX) // 2, (W - TILE_PX) // 2)]
            crops = np.stack([img[y:y + TILE_PX, x:x + TILE_PX] for y, x in tiles])
            hand = np.stack([hand_features(c) for c in crops])
            x = torch.from_numpy(crops).to(dev).permute(0, 3, 1, 2).float() / 255
            if GRAY:  # grayscale with per-tile fixed mean/contrast -> removes white balance & exposure
                g = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
                g = (g - g.mean((2, 3), keepdim=True)) / (g.std((2, 3), keepdim=True) + 1e-4)
                x = (0.45 + 0.2 * g).clamp(0, 1).expand(-1, 3, -1, -1)
            x = ((x - proc_mean) / proc_std).half()
            embs = []
            for i in range(0, len(x), 16):
                h = model(pixel_values=x[i:i + 16]).last_hidden_state
                embs.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu())
            emb = torch.cat(embs).numpy()
            sid = sample_id_train(f) if split == "train" else test_name_to_id(f)
            rec.append(dict(path=f, split=split, sample=sid, cam=camera_of(f), ntiles=len(tiles),
                            emb=emb.mean(0), emb_std=emb.std(0), hand=hand.mean(0), hand_std=hand.std(0)))
            print(f"{split} {sid:28s} {camera_of(f):24s} tiles={len(tiles):2d}", flush=True)
    np.savez(in_dir(CACHE_DIR, OUT), **{k: np.array([r[k] for r in rec]) for k in rec[0]})
    print("saved", OUT, "tile_mm=%.1f" % TILE_MM)


if __name__ == "__main__":
    main()
