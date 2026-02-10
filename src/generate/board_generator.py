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
import cv2

# === Config ===

CARD_DIR = Path(__file__).parent.parent.parent / "training_images"
BACKGROUNDS_DIR = Path(__file__).parent.parent.parent / "data" / "backgrounds"
NOISE_OBJECTS_DIR = Path(__file__).parent.parent.parent / "data" / "noise_objects"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "synthetic"


def get_noise_objects() -> List[Image.Image]:
    """Load all noise object images."""
    objects = []
    if NOISE_OBJECTS_DIR.exists():
        for path in NOISE_OBJECTS_DIR.glob("*.png"):
            try:
                img = Image.open(path).convert("RGBA")
                objects.append(img)
            except Exception:
                pass
    return objects


def add_noise_objects(
    board: Image.Image,
    placed_boxes: List[Tuple[int, int, int, int]],
    num_objects: int = 3,
) -> Image.Image:
    """Add random noise objects to the board, avoiding card positions."""
    objects = get_noise_objects()
    if not objects:
        return board
    
    width, height = board.size
    board = board.convert("RGBA")
    
    for _ in range(num_objects):
        obj = random.choice(objects).copy()
        
        # Random scale (0.5x to 1.5x)
        scale = random.uniform(0.5, 1.5)
        new_w = int(obj.width * scale)
        new_h = int(obj.height * scale)
        obj = obj.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Random rotation
        angle = random.uniform(-30, 30)
        obj = obj.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        
        ow, oh = obj.size
        
        # Try to place without overlapping cards
        for _ in range(20):
            x = random.randint(10, width - ow - 10) if width > ow + 20 else 10
            y = random.randint(10, height - oh - 10) if height > oh + 20 else 10
            
            bbox = (x, y, x + ow, y + oh)
            
            # Check overlap with cards
            overlaps = False
            for card_box in placed_boxes:
                if not (bbox[2] < card_box[0] or bbox[0] > card_box[2] or 
                        bbox[3] < card_box[1] or bbox[1] > card_box[3]):
                    overlaps = True
                    break
            
            if not overlaps:
                board.paste(obj, (x, y), obj)
                break
    
    return board.convert("RGB")


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


def apply_perspective_transform(
    image: Image.Image,
    annotations: List[dict],
    strength: float = 0.15,
) -> Tuple[Image.Image, List[dict]]:
    """
    Apply random perspective transform to simulate tilted camera angle.
    
    Args:
        image: PIL Image
        annotations: List of annotations with x_center, y_center, width, height (normalized)
        strength: How strong the perspective effect is (0-0.3 recommended)
    
    Returns:
        Transformed image and updated annotations
    """
    width, height = image.size
    
    # Convert to numpy for OpenCV
    img_array = np.array(image)
    
    # Define source points (corners of original image)
    src_pts = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])
    
    # Randomly perturb destination points to create perspective effect
    # This simulates viewing the table from different angles
    max_shift = int(min(width, height) * strength)
    
    # Random shifts for each corner (but keep it realistic - like camera tilt)
    # Top corners shift more than bottom (camera looking down at table)
    top_shift_x = random.randint(-max_shift, max_shift)
    top_shift_y = random.randint(0, max_shift)  # Only shift down/neutral
    bottom_shift_x = random.randint(-max_shift // 2, max_shift // 2)
    bottom_shift_y = random.randint(-max_shift // 2, 0)  # Only shift up/neutral
    
    dst_pts = np.float32([
        [top_shift_x, top_shift_y],  # top-left
        [width - top_shift_x, top_shift_y + random.randint(-max_shift//3, max_shift//3)],  # top-right
        [width - bottom_shift_x, height + bottom_shift_y],  # bottom-right
        [bottom_shift_x, height + bottom_shift_y + random.randint(-max_shift//3, max_shift//3)]  # bottom-left
    ])
    
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Compute inverse for mapping points back
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    
    # Apply transform to image
    transformed = cv2.warpPerspective(img_array, M, (width, height), 
                                       borderMode=cv2.BORDER_REPLICATE)
    
    # Update annotations - transform bounding box corners
    new_annotations = []
    for ann in annotations:
        # Get original bbox in pixel coordinates
        cx = ann['x_center'] * width
        cy = ann['y_center'] * height
        w = ann['width'] * width
        h = ann['height'] * height
        
        # Get corners of bbox
        corners = np.array([
            [cx - w/2, cy - h/2],  # top-left
            [cx + w/2, cy - h/2],  # top-right
            [cx + w/2, cy + h/2],  # bottom-right
            [cx - w/2, cy + h/2],  # bottom-left
        ], dtype=np.float32)
        
        # Transform corners
        corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
        transformed_corners = (M @ corners_homogeneous.T).T
        transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:3]
        
        # Compute new bounding box (axis-aligned)
        new_x_min = max(0, transformed_corners[:, 0].min())
        new_x_max = min(width, transformed_corners[:, 0].max())
        new_y_min = max(0, transformed_corners[:, 1].min())
        new_y_max = min(height, transformed_corners[:, 1].max())
        
        # Skip if bbox is too small or outside image
        new_w = new_x_max - new_x_min
        new_h = new_y_max - new_y_min
        if new_w < 20 or new_h < 20:
            continue
        if new_x_max <= 0 or new_x_min >= width or new_y_max <= 0 or new_y_min >= height:
            continue
        
        # Update annotation
        new_ann = ann.copy()
        new_ann['x_center'] = (new_x_min + new_w / 2) / width
        new_ann['y_center'] = (new_y_min + new_h / 2) / height
        new_ann['width'] = new_w / width
        new_ann['height'] = new_h / height
        new_annotations.append(new_ann)
    
    return Image.fromarray(transformed), new_annotations


def generate_board(
    card_paths: List[Tuple[Path, dict]],
    num_cards: int = 12,
    width: int = 1280,
    height: int = 960,
    max_attempts: int = 100,
    layout: str = "grid",  # "grid" or "random"
) -> Tuple[Image.Image, List[dict]]:
    """
    Generate a synthetic board image with cards.
    
    Args:
        layout: "grid" for realistic Set layout, "random" for scattered
    
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
    
    if layout == "grid":
        # Calculate grid layout to fill most of the image
        # Standard Set: 12 cards in 3x4 or 4x3 grid
        num_cards = len(selected)
        
        # Determine grid dimensions
        if num_cards <= 9:
            rows, cols = 3, 3
        elif num_cards <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4
        
        # Calculate card size to fill ~80% of the board
        padding = 30  # Edge padding
        card_gap = 15  # Gap between cards
        
        available_width = width - 2 * padding - (cols - 1) * card_gap
        available_height = height - 2 * padding - (rows - 1) * card_gap
        
        cell_width = available_width // cols
        cell_height = available_height // rows
        
        # Load a sample card to get aspect ratio
        sample_card = Image.open(selected[0][0])
        card_aspect = sample_card.width / sample_card.height
        
        # Calculate card size maintaining aspect ratio
        if cell_width / cell_height > card_aspect:
            card_h = int(cell_height * 0.9)  # 90% of cell
            card_w = int(card_h * card_aspect)
        else:
            card_w = int(cell_width * 0.9)
            card_h = int(card_w / card_aspect)
        
        # Place cards in grid with slight randomness
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= len(selected):
                    break
                
                card_path, label = selected[idx]
                idx += 1
                
                # Load and resize card
                card = Image.open(card_path).convert("RGBA")
                card = card.resize((card_w, card_h), Image.Resampling.LANCZOS)
                
                # Add slight rotation (-10 to +10 degrees)
                angle = random.uniform(-10, 10)
                rotated = card.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
                cw, ch = rotated.size
                
                # Calculate grid position with randomness
                base_x = padding + col * (cell_width + card_gap) + (cell_width - cw) // 2
                base_y = padding + row * (cell_height + card_gap) + (cell_height - ch) // 2
                
                # Add random offset (up to 10% of cell size)
                jitter_x = random.randint(-cell_width // 10, cell_width // 10)
                jitter_y = random.randint(-cell_height // 10, cell_height // 10)
                
                x = max(5, min(width - cw - 5, base_x + jitter_x))
                y = max(5, min(height - ch - 5, base_y + jitter_y))
                
                # Place the card
                board.paste(rotated, (x, y), rotated)
                
                # YOLO format annotation
                x_center = (x + cw / 2) / width
                y_center = (y + ch / 2) / height
                box_w = cw / width
                box_h = ch / height
                
                annotations.append({
                    "class_id": 0,
                    "card_class_id": label["class_id"],
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": box_w,
                    "height": box_h,
                    "label": label,
                })
                # Track placed boxes for noise object placement
                placed_boxes.append((x, y, x + cw, y + ch))
    else:
        # Random layout (original behavior but with larger cards)
        for card_path, label in selected:
            card = Image.open(card_path).convert("RGBA")
            
            # Larger scale for random layout (fill more space)
            angle = random.uniform(-20, 20)
            scale = random.uniform(1.2, 1.8)
            
            rotated = rotate_and_get_bbox(card, angle, scale)
            cw, ch = rotated.size
            
            placed = False
            for _ in range(max_attempts):
                pad = 20
                x = random.randint(pad, width - cw - pad) if width > cw + 2*pad else pad
                y = random.randint(pad, height - ch - pad) if height > ch + 2*pad else pad
                
                bbox = (x, y, x + cw, y + ch)
                
                if not check_overlap(bbox, placed_boxes, margin=20):
                    board.paste(rotated, (x, y), rotated)
                    placed_boxes.append(bbox)
                    
                    x_center = (x + cw / 2) / width
                    y_center = (y + ch / 2) / height
                    box_w = cw / width
                    box_h = ch / height
                    
                    annotations.append({
                        "class_id": 0,
                        "card_class_id": label["class_id"],
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": box_w,
                        "height": box_h,
                        "label": label,
                    })
                    placed = True
                    break
            
            if not placed:
                continue
    
    # Add noise objects (50% chance, 1-4 objects)
    if random.random() < 0.5:
        num_noise = random.randint(1, 4)
        board = add_noise_objects(board, placed_boxes, num_noise)
    
    # Apply random augmentations to entire board
    if random.random() < 0.3:
        board = board.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(board)
        board = enhancer.enhance(random.uniform(0.8, 1.2))
    
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(board)
        board = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Apply perspective transform (仿射变换) to simulate camera angle
    # Apply to ~70% of images for variety
    if random.random() < 0.7:
        strength = random.uniform(0.05, 0.15)  # Subtle to moderate perspective
        board, annotations = apply_perspective_transform(board, annotations, strength)
    
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
    num_cards_range: Tuple[int, int] = (9, 12),
    output_dir: Path = OUTPUT_DIR,
    width: int = 1280,
    height: int = 960,
    layout: str = "grid",  # "grid" or "random"
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
    print(f"Layout: {layout}, Cards per board: {num_cards_range[0]}-{num_cards_range[1]}")
    
    # Generate images
    for i in tqdm(range(num_images), desc="Generating boards"):
        num_cards = random.randint(*num_cards_range)
        
        # Mix layouts: 80% grid, 20% random for variety
        use_layout = layout if layout != "mixed" else ("grid" if random.random() < 0.8 else "random")
        
        board, annotations = generate_board(
            card_paths,
            num_cards=num_cards,
            width=width,
            height=height,
            layout=use_layout,
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
    parser.add_argument("--max-cards", type=int, default=12, help="Max cards per board")
    parser.add_argument("--width", type=int, default=1280, help="Image width")
    parser.add_argument("--height", type=int, default=960, help="Image height")
    parser.add_argument("--layout", type=str, default="mixed", choices=["grid", "random", "mixed"],
                        help="Layout style: grid (realistic), random (scattered), mixed (80/20)")
    args = parser.parse_args()
    
    generate_dataset(
        num_images=args.num,
        num_cards_range=(args.min_cards, args.max_cards),
        width=args.width,
        height=args.height,
        layout=args.layout,
    )


if __name__ == "__main__":
    main()
