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
OUTPUT_DIR = Path.home() / "data" / "set-solver" / "synthetic"


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


def _smooth_noise(width: int, height: int, scale: int = 8) -> np.ndarray:
    """Generate cheap Perlin-like noise via upscaled random arrays.

    Returns float32 array in [0, 1] of shape (height, width).
    """
    small_h = max(2, height // scale)
    small_w = max(2, width // scale)
    small = np.random.rand(small_h, small_w).astype(np.float32)
    big = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(big, 0.0, 1.0)


def _create_wood_background(width: int, height: int) -> Image.Image:
    """Procedural wood texture with grain lines and smooth noise."""
    # Brown / tan palette
    base_r = random.randint(120, 180)
    base_g = random.randint(70, 120)
    base_b = random.randint(30, 70)
    base = np.full((height, width, 3), [base_r, base_g, base_b], dtype=np.float32)

    # Sine-wave grain lines (horizontal)
    freq = random.uniform(0.02, 0.06)
    phase = random.uniform(0, 2 * math.pi)
    y_coords = np.arange(height).reshape(-1, 1).astype(np.float32)
    grain = np.sin(y_coords * freq + phase) * 15  # amplitude +-15
    base += grain[:, :, np.newaxis]

    # Multi-octave smooth noise for wood-knot variation
    for octave_scale in (32, 16, 8):
        noise = _smooth_noise(width, height, scale=octave_scale)
        base += (noise[:, :, np.newaxis] - 0.5) * 25

    pixels = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def _create_metal_background(width: int, height: int) -> Image.Image:
    """Procedural brushed-metal texture."""
    # Metallic gray palette
    base_v = random.randint(140, 200)
    base = np.full((height, width, 3), base_v, dtype=np.float32)

    # Horizontal brush streaks
    for _ in range(random.randint(60, 150)):
        y = random.randint(0, height - 1)
        thickness = random.randint(1, 3)
        intensity = random.uniform(-12, 12)
        y_end = min(height, y + thickness)
        base[y:y_end, :, :] += intensity

    # Smooth noise for subtle variation
    noise = _smooth_noise(width, height, scale=16)
    base += (noise[:, :, np.newaxis] - 0.5) * 20

    # Optional specular highlight (50% chance)
    if random.random() < 0.5:
        cx = random.uniform(0.2, 0.8) * width
        cy = random.uniform(0.2, 0.8) * height
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        highlight = np.exp(-dist / (max(width, height) * 0.4)) * 30
        base += highlight[:, :, np.newaxis]

    pixels = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def _create_glass_background(width: int, height: int) -> Image.Image:
    """Procedural glass/smooth surface texture."""
    # Pick two nearby colours for a gradient
    c1 = np.array([random.randint(180, 240)] * 3, dtype=np.float32)
    c2 = c1 + np.array([random.randint(-30, 30) for _ in range(3)], dtype=np.float32)
    c2 = np.clip(c2, 0, 255)

    # Vertical gradient
    t = np.linspace(0, 1, height, dtype=np.float32).reshape(-1, 1, 1)
    grad = c1 * (1 - t) + c2 * t
    base = np.broadcast_to(grad, (height, width, 3)).copy()

    # Reflection bands
    num_bands = random.randint(2, 5)
    for _ in range(num_bands):
        y_center = random.randint(0, height - 1)
        band_w = random.randint(height // 20, height // 6)
        yy = np.arange(height, dtype=np.float32)
        band = np.exp(-((yy - y_center) ** 2) / (2 * (band_w ** 2))) * random.uniform(10, 25)
        base += band.reshape(-1, 1, 1)

    # Minimal noise
    noise = _smooth_noise(width, height, scale=32)
    base += (noise[:, :, np.newaxis] - 0.5) * 8

    pixels = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


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

    # Procedural texture selection (weighted)
    r = random.random()
    if r < 0.35:
        return _create_wood_background(width, height)
    elif r < 0.55:
        return _create_metal_background(width, height)
    elif r < 0.75:
        return _create_glass_background(width, height)
    else:
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


def resize_card(card: Image.Image, target_w: int, target_h: int, is_striped: bool = False) -> Image.Image:
    """Resize card image, using gentler resampling for striped cards to avoid moire."""
    if is_striped:
        # Pre-blur to suppress high-frequency stripe ringing, then use BILINEAR
        card = card.filter(ImageFilter.GaussianBlur(radius=0.6))
        return card.resize((target_w, target_h), Image.Resampling.BILINEAR)
    return card.resize((target_w, target_h), Image.Resampling.LANCZOS)


def apply_card_rotation(card: Image.Image) -> Image.Image:
    """Rotate an RGBA card image by a random angle.

    Cards can be placed at any orientation on the table — some horizontal,
    some vertical, some diagonal.  Shear/perspective effects come from the
    whole-board perspective transform which simulates the camera angle.
    """
    angle = random.uniform(-180, 180)
    return card.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)


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


def _place_card(
    board: Image.Image,
    rotated: Image.Image,
    x: int,
    y: int,
    board_w: int,
    board_h: int,
    label: dict,
    annotations: List[dict],
    placed_boxes: List[Tuple[int, int, int, int]],
    min_visible: int = 20,
) -> bool:
    """Paste a card onto the board with edge-clipping support.

    Handles cards that extend beyond image boundaries by pasting only the
    visible portion and clipping the annotation bbox to image bounds.

    Returns True if the card was placed (visible area large enough).
    """
    cw, ch = rotated.size

    # Visible region in source-image coordinates
    src_x1 = max(0, -x)
    src_y1 = max(0, -y)
    src_x2 = min(cw, board_w - x)
    src_y2 = min(ch, board_h - y)

    visible_w = src_x2 - src_x1
    visible_h = src_y2 - src_y1
    if visible_w < min_visible or visible_h < min_visible:
        return False

    # Paste — crop source when it goes off-edge
    if src_x1 > 0 or src_y1 > 0 or src_x2 < cw or src_y2 < ch:
        visible = rotated.crop((src_x1, src_y1, src_x2, src_y2))
        board.paste(visible, (max(0, x), max(0, y)), visible)
    else:
        board.paste(rotated, (x, y), rotated)

    placed_boxes.append((x, y, x + cw, y + ch))

    # Annotation clipped to image bounds
    ax1 = max(0, x)
    ay1 = max(0, y)
    ax2 = min(board_w, x + cw)
    ay2 = min(board_h, y + ch)
    ann_w = ax2 - ax1
    ann_h = ay2 - ay1

    annotations.append({
        "class_id": 0,
        "card_class_id": label["class_id"],
        "x_center": (ax1 + ann_w / 2) / board_w,
        "y_center": (ay1 + ann_h / 2) / board_h,
        "width": ann_w / board_w,
        "height": ann_h / board_h,
        "label": label,
    })
    return True


def apply_perspective_transform(
    image: Image.Image,
    annotations: List[dict],
    strength: float = 0.15,
) -> Tuple[Image.Image, List[dict]]:
    """
    Apply random perspective transform to simulate tilted camera angle.

    Picks a random viewing direction (top/bottom/left/right) and creates a
    trapezoid warp where the "far" edge shrinks inward, like a real camera
    looking at a table from that side.

    Args:
        image: PIL Image
        annotations: List of annotations with x_center, y_center, width, height (normalized)
        strength: How strong the perspective effect is (0.1-0.35 recommended)

    Returns:
        Transformed image and updated annotations
    """
    width, height = image.size

    # Convert to numpy for OpenCV
    img_array = np.array(image)

    # Source corners: TL, TR, BR, BL
    src_pts = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    # Pick a random direction the camera is looking FROM.
    # The "far" edge (opposite side) shrinks inward to create a trapezoid.
    inset = int(min(width, height) * strength)
    direction = random.choice(["top", "bottom", "left", "right"])

    # Small random asymmetry so left/right insets aren't perfectly equal
    asym = random.uniform(0.7, 1.3)

    if direction == "bottom":
        # Camera looking from below — top edge (far) narrows
        inset_l = int(inset * asym)
        inset_r = int(inset / asym)
        dst_pts = np.float32([
            [inset_l, inset],          # TL moves right & down
            [width - inset_r, inset],   # TR moves left & down
            [width, height],             # BR stays
            [0, height]                  # BL stays
        ])
    elif direction == "top":
        # Camera looking from above — bottom edge (far) narrows
        inset_l = int(inset * asym)
        inset_r = int(inset / asym)
        dst_pts = np.float32([
            [0, 0],                          # TL stays
            [width, 0],                      # TR stays
            [width - inset_r, height - inset],  # BR moves left & up
            [inset_l, height - inset]           # BL moves right & up
        ])
    elif direction == "left":
        # Camera looking from the left — right edge (far) narrows
        inset_t = int(inset * asym)
        inset_b = int(inset / asym)
        dst_pts = np.float32([
            [0, 0],                          # TL stays
            [width - inset, inset_t],        # TR moves left & down
            [width - inset, height - inset_b],  # BR moves left & up
            [0, height]                      # BL stays
        ])
    else:  # right
        # Camera looking from the right — left edge (far) narrows
        inset_t = int(inset * asym)
        inset_b = int(inset / asym)
        dst_pts = np.float32([
            [inset, inset_t],                # TL moves right & down
            [width, 0],                      # TR stays
            [width, height],                 # BR stays
            [inset, height - inset_b]        # BL moves right & up
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
    layout: str = "grid",  # "grid", "random", "overlap", or "pile"
) -> Tuple[Image.Image, List[dict]]:
    """
    Generate a synthetic board image with cards.

    Args:
        layout: "grid" (realistic Set layout), "random" (scattered, no overlap),
                "overlap" (deliberate 10-40% overlap), "pile" (messy, heavy overlap)

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
            rows, cols = 3, 5

        # Calculate card size to fill most of the board
        padding = 15  # Edge padding
        card_gap = 8   # Gap between cards
        
        available_width = width - 2 * padding - (cols - 1) * card_gap
        available_height = height - 2 * padding - (rows - 1) * card_gap
        
        cell_width = available_width // cols
        cell_height = available_height // rows
        
        # Load a sample card to get aspect ratio
        sample_card = Image.open(selected[0][0])
        card_aspect = sample_card.width / sample_card.height
        
        # Calculate card size maintaining aspect ratio
        if cell_width / cell_height > card_aspect:
            card_h = int(cell_height * 0.95)  # 95% of cell
            card_w = int(card_h * card_aspect)
        else:
            card_w = int(cell_width * 0.95)
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
                is_striped = label["fill"] == 2  # partial/striped
                card = resize_card(card, card_w, card_h, is_striped=is_striped)

                # Per-card rotation only (shear/perspective applied to whole board)
                rotated = apply_card_rotation(card)
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
    elif layout == "random":
        # Random layout — scattered cards, no overlap, uniform size
        sample_card = Image.open(selected[0][0])
        card_aspect = sample_card.width / sample_card.height
        card_h = int(min(width, height) * 0.22)
        card_w = int(card_h * card_aspect)

        for card_path, label in selected:
            card = Image.open(card_path).convert("RGBA")
            is_striped = label["fill"] == 2
            card = resize_card(card, card_w, card_h, is_striped=is_striped)

            rotated = apply_card_rotation(card)
            cw, ch = rotated.size

            for _ in range(max_attempts):
                pad = 20
                x = random.randint(pad, width - cw - pad) if width > cw + 2 * pad else pad
                y = random.randint(pad, height - ch - pad) if height > ch + 2 * pad else pad

                bbox = (x, y, x + cw, y + ch)

                if not check_overlap(bbox, placed_boxes, margin=20):
                    _place_card(board, rotated, x, y, width, height,
                                label, annotations, placed_boxes)
                    break

    elif layout == "overlap":
        # Cards with deliberate 10-40% overlap — simulates crowded table
        sample_card = Image.open(selected[0][0])
        card_aspect = sample_card.width / sample_card.height
        card_h = int(min(width, height) * 0.22)
        card_w = int(card_h * card_aspect)

        for idx, (card_path, label) in enumerate(selected):
            card = Image.open(card_path).convert("RGBA")
            is_striped = label["fill"] == 2
            card = resize_card(card, card_w, card_h, is_striped=is_striped)

            rotated = apply_card_rotation(card)
            cw, ch = rotated.size

            if idx == 0 or not placed_boxes:
                # First card: place in central area
                x = random.randint(width // 6, max(width // 6 + 1, width - cw - width // 6))
                y = random.randint(height // 6, max(height // 6 + 1, height - ch - height // 6))
            else:
                # Pick a random existing card to overlap with
                target_box = random.choice(placed_boxes)
                tx1, ty1, tx2, ty2 = target_box
                tw, th = tx2 - tx1, ty2 - ty1

                overlap_frac = random.uniform(0.1, 0.4)
                side = random.choice(["left", "right", "top", "bottom"])
                if side == "right":
                    x = tx2 - int(cw * overlap_frac)
                    y = ty1 + random.randint(-ch // 3, max(1, th - ch // 3))
                elif side == "left":
                    x = tx1 - cw + int(cw * overlap_frac)
                    y = ty1 + random.randint(-ch // 3, max(1, th - ch // 3))
                elif side == "bottom":
                    x = tx1 + random.randint(-cw // 3, max(1, tw - cw // 3))
                    y = ty2 - int(ch * overlap_frac)
                else:  # top
                    x = tx1 + random.randint(-cw // 3, max(1, tw - cw // 3))
                    y = ty1 - ch + int(ch * overlap_frac)

                # Allow slight edge clipping
                x = max(-cw // 5, min(width - cw * 4 // 5, x))
                y = max(-ch // 5, min(height - ch * 4 // 5, y))

            _place_card(board, rotated, x, y, width, height,
                        label, annotations, placed_boxes)

    elif layout == "pile":
        # Messy scattered layout — no overlap avoidance, variable card sizes
        sample_card = Image.open(selected[0][0])
        card_aspect = sample_card.width / sample_card.height
        base_card_h = int(min(width, height) * 0.22)

        for card_path, label in selected:
            # Variable card size ±15%
            scale_var = random.uniform(0.85, 1.15)
            card_h = int(base_card_h * scale_var)
            card_w = int(card_h * card_aspect)

            card = Image.open(card_path).convert("RGBA")
            is_striped = label["fill"] == 2
            card = resize_card(card, card_w, card_h, is_striped=is_striped)

            rotated = apply_card_rotation(card)
            cw, ch = rotated.size

            # Random position — allow partial off-edge
            margin = cw // 5
            x = random.randint(-margin, width - cw + margin)
            y = random.randint(-margin, height - ch + margin)

            _place_card(board, rotated, x, y, width, height,
                        label, annotations, placed_boxes)
    
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
    
    # Apply perspective transform (透视变换) to simulate camera angle
    # Apply to ~80% of images; remaining 20% stay top-down for variety
    if random.random() < 0.8:
        strength = random.uniform(0.10, 0.30)  # Noticeable camera tilt
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
    num_cards_range: Tuple[int, int] = (3, 18),
    output_dir: Path = OUTPUT_DIR,
    width: int = 1280,
    height: int = 960,
    layout: str = "mixed",
    val_fraction: float = 0.2,
):
    """Generate a full synthetic dataset with proper train/val split."""
    output_dir = Path(output_dir)

    # Create train/val directory structure
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Load all card paths
    card_paths = get_all_card_paths()
    print(f"Loaded {len(card_paths)} card images")
    print(f"Layout: {layout}, Cards per board: {num_cards_range[0]}-{num_cards_range[1]}")

    num_val = int(num_images * val_fraction)
    num_train = num_images - num_val
    print(f"Split: {num_train} train, {num_val} val")

    # Generate images
    for i in tqdm(range(num_images), desc="Generating boards"):
        num_cards = random.randint(*num_cards_range)

        # Determine layout: 35% grid, 25% random, 25% overlap, 15% pile
        if layout == "mixed":
            r = random.random()
            if r < 0.35:
                use_layout = "grid"
            elif r < 0.60:
                use_layout = "random"
            elif r < 0.85:
                use_layout = "overlap"
            else:
                use_layout = "pile"
        else:
            use_layout = layout

        board, annotations = generate_board(
            card_paths,
            num_cards=num_cards,
            width=width,
            height=height,
            layout=use_layout,
        )

        # Deterministic split: first num_train → train, rest → val
        split = "train" if i < num_train else "val"

        # Save image
        img_path = output_dir / "images" / split / f"board_{i:05d}.png"
        board.save(img_path)

        # Save YOLO annotation
        ann_path = output_dir / "labels" / split / f"board_{i:05d}.txt"
        save_yolo_annotation(annotations, ann_path)

        # Save detailed annotation (for classifier training later)
        json_path = output_dir / "labels" / split / f"board_{i:05d}.json"
        with open(json_path, "w") as f:
            json.dump(annotations, f)

    # Create dataset YAML for YOLO training
    yaml_content = f"""path: {output_dir.absolute()}
train: images/train
val: images/val

names:
  0: card
"""
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\nGenerated {num_images} images ({num_train} train, {num_val} val)")
    print(f"Dataset YAML: {output_dir / 'dataset.yaml'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic board images")
    parser.add_argument("--num", type=int, default=5000, help="Number of images to generate")
    parser.add_argument("--min-cards", type=int, default=3, help="Min cards per board")
    parser.add_argument("--max-cards", type=int, default=18, help="Max cards per board")
    parser.add_argument("--width", type=int, default=1280, help="Image width")
    parser.add_argument("--height", type=int, default=960, help="Image height")
    parser.add_argument("--layout", type=str, default="mixed",
                        choices=["grid", "random", "overlap", "pile", "mixed"],
                        help="Layout style")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of images for validation (default: 0.2)")
    args = parser.parse_args()

    generate_dataset(
        num_images=args.num,
        num_cards_range=(args.min_cards, args.max_cards),
        width=args.width,
        height=args.height,
        layout=args.layout,
        val_fraction=args.val_fraction,
    )


if __name__ == "__main__":
    main()
