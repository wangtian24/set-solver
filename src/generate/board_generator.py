"""
Generate synthetic board images for detection training.

Takes cropped card images and composites them onto backgrounds
with random positions, rotations, scales. Outputs YOLO-format annotations.
"""

import random
import math
from pathlib import Path
from typing import List, Tuple, Optional
import json

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm

# === Config ===

CARD_DIR = Path(__file__).parent.parent.parent / "training_images"
BACKGROUNDS_DIR = Path(__file__).parent.parent.parent / "data" / "backgrounds"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "synthetic"


def get_all_card_paths() -> List[Tuple[Path, dict]]:
    """Get all card image paths with their labels."""
    cards = []
    
    number_map = {"one": 0, "two": 1, "three": 2}
    color_map = {"red": 0, "green": 1, "blue": 2}
    shape_map = {"diamond": 0, "oval": 1, "squiggle": 2}
    fill_map = {"empty": 0, "full": 1, "partial": 2}
    
    for number in number_map:
        for color in color_map:
            for shape in shape_map:
                for fill in fill_map:
                    folder = CARD_DIR / number / color / shape / fill
                    if folder.exists():
                        for img_path in folder.glob("*.png"):
                            label = {
                                "number": number_map[number],
                                "color": color_map[color],
                                "shape": shape_map[shape],
                                "fill": fill_map[fill],
                            }
                            # Compute class ID (0-80)
                            class_id = (
                                label["number"] * 27 +
                                label["color"] * 9 +
                                label["shape"] * 3 +
                                label["fill"]
                            )
                            label["class_id"] = class_id
                            cards.append((img_path, label))
    
    return cards


def create_solid_background(width: int, height: int) -> Image.Image:
    """Create a solid color background (table-like colors)."""
    colors = [
        (34, 139, 34),    # Forest green (card table)
        (0, 100, 0),      # Dark green
        (139, 69, 19),    # Saddle brown (wood)
        (160, 82, 45),    # Sienna
        (245, 245, 220),  # Beige
        (211, 211, 211),  # Light gray
        (47, 79, 79),     # Dark slate gray
        (72, 61, 139),    # Dark slate blue
    ]
    color = random.choice(colors)
    # Add some noise
    img = Image.new("RGB", (width, height), color)
    pixels = np.array(img)
    noise = np.random.randint(-15, 15, pixels.shape, dtype=np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def load_background(width: int, height: int) -> Image.Image:
    """Load a background image or create synthetic one."""
    if BACKGROUNDS_DIR.exists():
        bg_files = list(BACKGROUNDS_DIR.glob("*.jpg")) + list(BACKGROUNDS_DIR.glob("*.png"))
        if bg_files:
            bg_path = random.choice(bg_files)
            bg = Image.open(bg_path).convert("RGB")
            # Resize/crop to target size
            bg = bg.resize((width, height), Image.Resampling.LANCZOS)
            return bg
    
    # Fall back to solid color
    return create_solid_background(width, height)


def rotate_and_get_bbox(
    card: Image.Image,
    angle: float,
    scale: float,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Rotate card and compute its bounding box.
    
    Returns rotated image (with alpha) and bbox (x1, y1, x2, y2) relative to image.
    """
    # Scale the card
    w, h = card.size
    new_w, new_h = int(w * scale), int(h * scale)
    card = card.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Rotate with expand to fit
    rotated = card.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    
    return rotated


def check_overlap(
    new_box: Tuple[int, int, int, int],
    existing_boxes: List[Tuple[int, int, int, int]],
    margin: int = 10,
) -> bool:
    """Check if new_box overlaps with any existing boxes."""
    x1, y1, x2, y2 = new_box
    
    for ex1, ey1, ex2, ey2 in existing_boxes:
        # Add margin
        ex1 -= margin
        ey1 -= margin
        ex2 += margin
        ey2 += margin
        
        # Check overlap
        if not (x2 < ex1 or x1 > ex2 or y2 < ey1 or y1 > ey2):
            return True
    
    return False


def generate_board(
    card_paths: List[Tuple[Path, dict]],
    num_cards: int = 12,
    width: int = 1280,
    height: int = 960,
    max_attempts: int = 100,
) -> Tuple[Image.Image, List[dict]]:
    """
    Generate a synthetic board image with cards.
    
    Returns:
        - Board image
        - List of annotations (bbox + class_id for each card)
    """
    # Create background
    board = load_background(width, height)
    
    # Select random cards
    selected = random.sample(card_paths, min(num_cards, len(card_paths)))
    
    annotations = []
    placed_boxes = []
    
    for card_path, label in selected:
        # Load card
        card = Image.open(card_path).convert("RGBA")
        
        # Random transformations
        angle = random.uniform(0, 360)
        scale = random.uniform(0.4, 0.8)
        
        # Rotate card
        rotated = rotate_and_get_bbox(card, angle, scale)
        cw, ch = rotated.size
        
        # Try to place without overlap
        placed = False
        for _ in range(max_attempts):
            # Random position (with padding from edges)
            pad = 20
            x = random.randint(pad, width - cw - pad) if width > cw + 2*pad else pad
            y = random.randint(pad, height - ch - pad) if height > ch + 2*pad else pad
            
            # Bounding box
            bbox = (x, y, x + cw, y + ch)
            
            if not check_overlap(bbox, placed_boxes):
                # Place the card
                board.paste(rotated, (x, y), rotated)
                placed_boxes.append(bbox)
                
                # YOLO format: class_id, x_center, y_center, width, height (normalized)
                x_center = (x + cw / 2) / width
                y_center = (y + ch / 2) / height
                box_w = cw / width
                box_h = ch / height
                
                annotations.append({
                    "class_id": 0,  # Single class: "card" for detection
                    "card_class_id": label["class_id"],  # 81-class for classification
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": box_w,
                    "height": box_h,
                    "label": label,
                })
                placed = True
                break
        
        if not placed:
            # Skip this card if can't place
            continue
    
    # Apply random augmentations to entire board
    if random.random() < 0.3:
        # Slight blur
        board = board.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    if random.random() < 0.5:
        # Brightness adjustment
        enhancer = ImageEnhance.Brightness(board)
        board = enhancer.enhance(random.uniform(0.8, 1.2))
    
    if random.random() < 0.5:
        # Contrast adjustment
        enhancer = ImageEnhance.Contrast(board)
        board = enhancer.enhance(random.uniform(0.8, 1.2))
    
    return board, annotations


def save_yolo_annotation(annotations: List[dict], output_path: Path):
    """Save annotations in YOLO format."""
    with open(output_path, "w") as f:
        for ann in annotations:
            # YOLO format: class_id x_center y_center width height
            line = f"0 {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
            f.write(line)


def generate_dataset(
    num_images: int = 5000,
    num_cards_range: Tuple[int, int] = (9, 18),
    output_dir: Path = OUTPUT_DIR,
    width: int = 1280,
    height: int = 960,
):
    """Generate a full synthetic dataset."""
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all card paths
    card_paths = get_all_card_paths()
    print(f"Loaded {len(card_paths)} card images")
    
    # Generate images
    for i in tqdm(range(num_images), desc="Generating boards"):
        num_cards = random.randint(*num_cards_range)
        
        board, annotations = generate_board(
            card_paths,
            num_cards=num_cards,
            width=width,
            height=height,
        )
        
        # Save image
        img_path = images_dir / f"board_{i:05d}.jpg"
        board.save(img_path, quality=90)
        
        # Save YOLO annotation
        ann_path = labels_dir / f"board_{i:05d}.txt"
        save_yolo_annotation(annotations, ann_path)
        
        # Save detailed annotation (for classifier training later)
        json_path = labels_dir / f"board_{i:05d}.json"
        with open(json_path, "w") as f:
            json.dump(annotations, f)
    
    # Create dataset YAML for YOLO training
    yaml_content = f"""
path: {output_dir.absolute()}
train: images
val: images

names:
  0: card
"""
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    print(f"\nGenerated {num_images} images in {output_dir}")
    print(f"Dataset YAML: {output_dir / 'dataset.yaml'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic board images")
    parser.add_argument("--num", type=int, default=5000, help="Number of images to generate")
    parser.add_argument("--min-cards", type=int, default=9, help="Min cards per board")
    parser.add_argument("--max-cards", type=int, default=18, help="Max cards per board")
    parser.add_argument("--width", type=int, default=1280, help="Image width")
    parser.add_argument("--height", type=int, default=960, help="Image height")
    args = parser.parse_args()
    
    generate_dataset(
        num_images=args.num,
        num_cards_range=(args.min_cards, args.max_cards),
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
