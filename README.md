# Suspicious Photo Detection

Per-outlet unsupervised suspicious photo detection.

## Approach

DINOv2 ViT-S/14, 3-region crops (center + top-half + bottom-half) concatenated to a 1152-d descriptor, L2-normalized, Euclidean distance to the folder's geometric median, MAD modified z-score, cutoff 2.50. The cutoff is absolute so clean folders return an empty `flagged_images` list.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads DINOv2 weights (~85 MB, requires internet once) and computes embeddings for every image. Both are cached, so subsequent runs are fast.

## Run

```bash
python app/pipeline_multiregion.py
```

Expects a `dataset/` folder containing `outlet_*` subfolders at the same level as `app/`.

Tests: `pytest tests/`

## Output

Two files are written to `output/`:

- **`results_multiregion.json`** — one record per outlet in the assignment schema:
  ```json
  {
    "outlet_id": "outlet_xxxx",
    "total_images": 8,
    "flagged_images": [
      {
        "file_name": "image_0001.jpg",
        "suspicion_score": 0.8015,
        "reason": "Low similarity to the outlet's other 7 photos (multi-region anomaly score 3.9, beyond the folder's normal range)."
      }
    ],
    "ranking": ["image_0001.jpg", "image_0008.jpg", "..."]
  }
  ```
- **`results_multiregion.csv`** — one row per image with columns `outlet_id`, `total_images`, `file_name`, `rank`, `suspicion_score`, `modified_zscore`, `flagged`, `reason`.

Full rationale, trade-offs, and limitations: WRITEUP.md
