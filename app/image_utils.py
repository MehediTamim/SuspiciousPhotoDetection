import sys
from collections import Counter
from pathlib import Path

from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def detect_image_size(dataset_dir: Path) -> tuple[int, int]:
    for path in sorted(dataset_dir.rglob("*")):
        if path.suffix.lower() in EXTENSIONS:
            with Image.open(path) as img:
                return img.size  # (width, height)
    raise FileNotFoundError(f"No images found under {dataset_dir}")


def main():
    if len(sys.argv) > 1:
        dataset_dir = Path(sys.argv[1])
    else:
        dataset_dir = Path(__file__).resolve().parent.parent / "dataset"

    paths = sorted(p for p in dataset_dir.rglob("*") if p.suffix.lower() in EXTENSIONS)

    if not paths:
        print("No images found under", dataset_dir)
        sys.exit(1)

    size_counter: Counter = Counter()
    for p in paths:
        with Image.open(p) as img:
            size_counter[img.size] += 1

    print(f"Total images scanned : {len(paths)}")
    print(f"Distinct sizes found : {len(size_counter)}\n")

    for (w, h), count in size_counter.most_common():
        print(f"  {w}x{h}  —  {count} image(s)")

    if len(size_counter) == 1:
        (w, h) = next(iter(size_counter))
        print(f"\nAll images are {w}x{h}. IMG_W={w}, IMG_H={h} is correct.")
        sys.exit(0)
    else:
        dominant_w, dominant_h = size_counter.most_common(1)[0][0]
        print(f"\nSizes are NOT uniform. Dominant size: {dominant_w}x{dominant_h}.")
        print("Review the pipeline's region-split logic for non-standard images.")
        sys.exit(1)


if __name__ == "__main__":
    main()
