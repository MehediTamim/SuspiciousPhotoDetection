import csv
import glob
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from image_utils import detect_image_size  # local utility

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / "cache" / "embeddings_multiregion"

MODEL_NAME = "dinov2_vits14"  # 384-d CLS embedding
RESIZE_SIZE = 256  # shorter-side resize, official eval recipe
CROP_SIZE = 224  # must be a multiple of the patch size (14)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 4
NUM_THREADS = 12
MAD_CUTOFF = 2.50
MAD_SCALE = 0.6745
USE_CACHE = True

torch.set_num_threads(NUM_THREADS)
def _base_transform():
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _region_transform(top_rows, bottom_rows):
    def crop_fn(img):
        return img.crop((0, top_rows, img.size[0], bottom_rows))
    return transforms.Compose([
        transforms.Lambda(crop_fn),
        transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


TF_CENTER = _base_transform()


class Embedder:
    def __init__(self, tf_top, tf_bottom):
        self.model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
        self.model.eval()
        self.tf_top = tf_top
        self.tf_bottom = tf_bottom

    @torch.inference_mode()
    def _embed_with(self, files, tf):
        vecs = []
        for i in range(0, len(files), BATCH_SIZE):
            batch = files[i:i + BATCH_SIZE]
            x = torch.stack([tf(Image.open(f).convert("RGB")) for f in batch])
            vecs.append(self.model(x).cpu().numpy())
        return np.concatenate(vecs, axis=0).astype(np.float32)

    def embed(self, files):
        c = self._embed_with(files, TF_CENTER)
        t = self._embed_with(files, self.tf_top)
        b = self._embed_with(files, self.tf_bottom)
        emb = np.concatenate([c, t, b], axis=1)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return emb


def embed_outlet(embedder, outlet_dir):
    files = sorted(glob.glob(os.path.join(outlet_dir, "*.jpg")))
    names = [os.path.basename(f) for f in files]
    cache_path = CACHE_DIR / f"{os.path.basename(outlet_dir)}.npz"

    if USE_CACHE and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        if list(data["names"]) == names:
            return names, data["emb"]

    emb = embedder.embed(files) if files else np.empty((0, 0), dtype=np.float32)
    if USE_CACHE and files:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, names=np.array(names), emb=emb)
    return names, emb


def geometric_median(X, eps=1e-8, iters=200):
    # Weiszfeld iteration. The median has a 50% breakdown point, so fake images
    # cannot drag the folder center toward themselves the way they would a mean.
    y = X.mean(axis=0)
    for _ in range(iters):
        d = np.clip(np.linalg.norm(X - y, axis=1), eps, None)
        w = 1.0 / d
        y_new = (X * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            break
        y = y_new
    return y


def modified_zscores(dist):
    med = np.median(dist)
    mad = np.median(np.abs(dist - med))
    if mad == 0:
        # degenerate case: identical distances; fall back to plain std
        scale = dist.std()
        return np.zeros_like(dist) if scale == 0 else (dist - med) / scale
    return MAD_SCALE * (dist - med) / mad


def suspicion_from_modz(modz):
    # sigmoid centered at the cutoff, so 0.5 sits exactly on the flag line
    return 1.0 / (1.0 + np.exp(-(modz - MAD_CUTOFF)))


def make_reason(modz, n_others):
    return (f"Low similarity to the outlet's other {n_others} photos "
            f"(multi-region anomaly score {modz:.1f}, beyond the folder's normal range).")


def analyze_outlet(outlet_id, names, emb):
    n = len(names)
    center = geometric_median(emb)
    dist = np.linalg.norm(emb - center, axis=1)
    modz = modified_zscores(dist)
    scores = suspicion_from_modz(modz)
    order = np.argsort(-dist)

    flagged = [
        {
            "file_name": names[i],
            "suspicion_score": round(float(scores[i]), 4),
            "reason": make_reason(modz[i], n - 1),
        }
        for i in order if modz[i] > MAD_CUTOFF
    ]
    record = {
        "outlet_id": outlet_id,
        "total_images": n,
        "flagged_images": flagged,
        "ranking": [names[i] for i in order],
    }
    rows = [
        {
            "outlet_id": outlet_id,
            "total_images": n,
            "file_name": names[i],
            "rank": r + 1,
            "suspicion_score": round(float(scores[i]), 4),
            "modified_zscore": round(float(modz[i]), 3),
            "flagged": bool(modz[i] > MAD_CUTOFF),
            "reason": make_reason(modz[i], n - 1) if modz[i] > MAD_CUTOFF else "",
        }
        for r, i in enumerate(order)
    ]
    return record, rows


def write_outputs(records, rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "results_multiregion.json", "w") as f:
        json.dump(records, f, indent=2)
    fields = ["outlet_id", "total_images", "file_name", "rank",
              "suspicion_score", "modified_zscore", "flagged", "reason"]
    with open(OUTPUT_DIR / "results_multiregion.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    img_w, img_h = detect_image_size(DATASET_DIR)
    mid_row = img_h // 2
    tf_top = _region_transform(0, mid_row)
    tf_bottom = _region_transform(mid_row, img_h)

    outlets = sorted(glob.glob(str(DATASET_DIR / "outlet_*")))
    print(f"found {len(outlets)} outlets")
    print(f"image size: {img_w}x{img_h}  mad cutoff: {MAD_CUTOFF}")
    embedder = Embedder(tf_top, tf_bottom)

    records, rows, total_flagged = [], [], 0
    for k, outlet_dir in enumerate(outlets, 1):
        outlet_id = os.path.basename(outlet_dir)
        names, emb = embed_outlet(embedder, outlet_dir)
        if not names:
            records.append({"outlet_id": outlet_id, "total_images": 0,
                            "flagged_images": [], "ranking": []})
            continue
        record, per_image = analyze_outlet(outlet_id, names, emb)
        records.append(record)
        rows.extend(per_image)
        total_flagged += len(record["flagged_images"])
        print(f"  [{k:3d}/{len(outlets)}] {outlet_id}  "
              f"n={len(names):2d}  flagged={len(record['flagged_images'])}")

    write_outputs(records, rows)
    print(f"\ndone. {len(records)} outlets, {total_flagged} images flagged.")

if __name__ == "__main__":
    main()