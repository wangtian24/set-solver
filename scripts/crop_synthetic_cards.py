#!/usr/bin/env python3
"""
Crop individual cards from synthetic board images using ground truth annotations.

Produces card images that reflect real inference conditions: perspective distortion,
rotation, compression artifacts, edge clipping from imperfect bounding boxes.

Output goes to training_images_synthetic/ with the same folder structure as
training_images/ (number/color/shape/fill/*.png).
"""

import json
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
SYNTHETIC_DIR = Path.home() / "data" / "set-solver" / "synthetic"
OUTPUT_DIR = PROJECT_ROOT / "training_images_synthetic"

NUMBER_NAMES = ["one", "two", "three"]
COLOR_NAMES = ["red", "green", "blue"]
SHAPE_NAMES = ["diamond", "oval", "squiggle"]
FILL_NAMES = ["empty", "full", "partial"]


def crop_cards(max_boards: int = None):
    images_dir = SYNTHETIC_DIR / "images"
    labels_dir = SYNTHETIC_DIR / "labels"

    all_images = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if max_boards:
        all_images = all_images[:max_boards]

    count = 0
    for img_path in tqdm(all_images, desc="Cropping cards"):
        stem = img_path.stem
        json_path = labels_dir / f"{stem}.json"
        if not json_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        with open(json_path) as f:
            annotations = json.load(f)

        for ann in annotations:
            label = ann["label"]
            number = NUMBER_NAMES[label["number"]]
            color = COLOR_NAMES[label["color"]]
            shape = SHAPE_NAMES[label["shape"]]
            fill = FILL_NAMES[label["fill"]]

            # Convert normalized coords to pixel bbox
            cx = ann["x_center"] * w
            cy = ann["y_center"] * h
            bw = ann["width"] * w
            bh = ann["height"] * h

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            # Add small random bbox jitter (simulate imperfect detection)
            jitter = max(bw, bh) * random.uniform(0, 0.05)
            x1 += random.uniform(-jitter, jitter)
            y1 += random.uniform(-jitter, jitter)
            x2 += random.uniform(-jitter, jitter)
            y2 += random.uniform(-jitter, jitter)

            # Clamp to image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            crop = img.crop((x1, y1, x2, y2))

            # Save to folder structure
            out_dir = OUTPUT_DIR / number / color / shape / fill
            out_dir.mkdir(parents=True, exist_ok=True)
            crop.save(out_dir / f"syn_{stem}_{count}.png")
            count += 1

    print(f"\nCropped {count} card images to {OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop cards from synthetic boards")
    parser.add_argument("--max-boards", type=int, default=None, help="Limit number of boards")
    args = parser.parse_args()

    crop_cards(max_boards=args.max_boards)
