"""H18: second embedding family (SigLIP, image-text) + DINOv2 mid-layer, for late fusion and concept axes.

Rationale (compositional generalisation, see RESEARCH_LOG H18): directions that two differently trained
encoders agree on are more likely real grain-size signal than one model's artefacts; disagreement on a
test photo measures out-of-domain uncertainty; SigLIP's text-aligned space gives *named* factor axes
(loose vs clumped, wet vs dry, clean vs silty) usable for routing and for removing nuisance factors.

Outputs cache/multi_embed.npz:
  sand_*  : SandSnap v2 + test photos at 12 px/mm, 224 px tiles (same tiles as sand_features.py)
            siglip (1152, RGB, per-tile L2-normalised then averaged), dino_mid (768, DINOv2 layer 6
            mean-patch, grayscale-normalised input)
  comp_*  : competition train + test photos at 4.6 px/mm, 73 mm tiles (features.py tiling): siglip
  text_*  : concept prompt embeddings (L2-normalised, averaged over templates)
SigLIP-so400m-patch14-384 (Apache-2.0) is loaded from a local copy (HF download hit an httpx/brotli
bug); the fast tokenizer is built from tokenizer.json to avoid adding sentencepiece to myenv.
usage: python src/multi_embed.py
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedTokenizerFast

from common import CACHE_DIR, camera_of, effective_ppm, list_images, sample_id_train, test_name_to_id
from features import TILE_PX, load_scaled, valid_tiles
from sand_features import SANDSNAP, load, tiles

SIGLIP = "/scratch/m25csa007/models/siglip-so400m-patch14-384"
dev = "cuda"
CONCEPTS = {
    "loose_vs_clumped": (["loose sand grains", "a loose granular sand surface", "uncompacted loose sand"],
                         ["dried clumped soil aggregates", "crumbly soil clods", "dry lumpy soil with aggregates"]),
    "wet_vs_dry": (["wet sand", "moist damp soil", "wet soil surface"],
                   ["dry sand", "dry dusty soil", "dry soil surface"]),
    "clean_vs_silty": (["clean sand without silt or clay", "washed clean quartz sand", "well-sorted clean sand"],
                       ["silty clayey sand with fine particles", "muddy sand with silt and clay", "sandy silt loam"]),
    "gravelly_vs_uniform": (["sand with gravel and pebbles", "gravelly soil with stones", "pebbles mixed in sand"],
                            ["uniform sand without pebbles", "homogeneous fine material", "sand with no stones"]),
    "coarse_vs_fine": (["coarse sand", "coarse grained sand", "very coarse sand grains"],
                       ["fine sand", "very fine sand", "fine grained silt"]),
}
TEMPLATES = ["a close-up photo of {}.", "a top-down photograph of {}.", "{}"]


@torch.no_grad()
def siglip_image(model, crops):
    x = torch.from_numpy(crops).to(dev).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(384, 384), mode="bicubic", align_corners=False).clamp(0, 1)
    out = []
    for i in range(0, len(x), 16):
        f = model.get_image_features(pixel_values=((x[i:i + 16] - 0.5) / 0.5).half())
        out.append(F.normalize(getattr(f, "pooler_output", f).float(), dim=-1))
    return torch.cat(out).mean(0).cpu().numpy()


@torch.no_grad()
def dino_mid(model, crops, layer=6):
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    x = torch.from_numpy(crops).to(dev).permute(0, 3, 1, 2).float() / 255
    g = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
    g = (g - g.mean((2, 3), keepdim=True)) / (g.std((2, 3), keepdim=True) + 1e-4)
    x = (((0.45 + 0.2 * g).clamp(0, 1).expand(-1, 3, -1, -1) - mean) / std).half()
    hs = model(pixel_values=x, output_hidden_states=True).hidden_states[layer]
    return hs[:, 1:].mean(1).float().mean(0).cpu().numpy()


@torch.no_grad()
def main():
    sig = AutoModel.from_pretrained(SIGLIP).to(dev).eval().half()
    sp = json.load(open(f"{SIGLIP}/special_tokens_map.json"))
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{SIGLIP}/tokenizer.json",
                                  **{k: (v["content"] if isinstance(v, dict) else v) for k, v in sp.items()})
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval().half()
    out = {}

    # ---- text concept embeddings ----
    for name, (pos, neg) in CONCEPTS.items():
        for side, phrases in (("pos", pos), ("neg", neg)):
            prompts = [t.format(p) for p in phrases for t in TEMPLATES]
            ids = tok(prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")["input_ids"].to(dev)
            f = sig.get_text_features(input_ids=ids)
            f = F.normalize(getattr(f, "pooler_output", f).float(), dim=-1).mean(0)
            out[f"text_{name}_{side}"] = F.normalize(f, dim=0).cpu().numpy()

    # ---- sand scale: SandSnap v2 + test ----
    idx = pd.read_csv(f"{SANDSNAP}/sandsnap_index.csv")
    items = [("sandsnap", f"{SANDSNAP}/{r.local}", 1 / r.mm_px, r.sample_key, int(r.dry)) for r in idx.itertuples()]
    tr, te = list_images()
    items += [("test", f, effective_ppm(f), test_name_to_id(f), -1) for f in te]
    rec = []
    for src, path, ppm, key, dry in items:
        try:
            T = tiles(load(path, ppm))
        except ValueError:
            continue
        if T is None:
            continue
        rec.append(dict(src=src, path=path, key=key, dry=dry, siglip=siglip_image(sig, T), dino_mid=dino_mid(dino, T)))
    for k in rec[0]:
        out[f"sand_{k}"] = np.array([r[k] for r in rec])
    print("sand-scale images:", {s: int((out['sand_src'] == s).sum()) for s in ("sandsnap", "test")}, flush=True)

    # ---- competition scale: train + test at 4.6 px/mm ----
    rec = []
    for split, files in (("train", tr), ("test", te)):
        for f in files:
            img = load_scaled(f)
            t = valid_tiles(img)
            crops = np.stack([img[y:y + TILE_PX, x:x + TILE_PX] for y, x in t])
            sid = sample_id_train(f) if split == "train" else test_name_to_id(f)
            rec.append(dict(split=split, path=f, sample=sid, cam=camera_of(f), siglip=siglip_image(sig, crops)))
    for k in rec[0]:
        out[f"comp_{k}"] = np.array([r[k] for r in rec])
    np.savez(f"{CACHE_DIR}/multi_embed.npz", **out)
    print("saved cache/multi_embed.npz", {k: v.shape for k, v in out.items() if k.endswith(("siglip", "dino_mid"))})


if __name__ == "__main__":
    main()
