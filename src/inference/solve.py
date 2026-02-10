"""
End-to-end Set solver pipeline.

Photo → Detect cards → Classify each → Find Sets → Visualize
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.train.classifier import (
    SetCardClassifier,
    NUMBER_NAMES, COLOR_NAMES, SHAPE_NAMES, FILL_NAMES,
)
from src.solver.set_finder import Card, Shape, Color, Number, Fill, find_all_sets


WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"

# Chinese shorthand names: {1,2,3}-{实，空，线}-{红，绿，紫}-{菱，圆，弯}
CHINESE_NUMBER = {"one": "1", "two": "2", "three": "3"}
CHINESE_FILL = {"full": "实", "empty": "空", "partial": "线"}
CHINESE_COLOR = {"red": "红", "green": "绿", "blue": "紫"}
CHINESE_SHAPE = {"diamond": "菱", "oval": "圆", "squiggle": "弯"}


def card_to_chinese(attrs: dict) -> str:
    """Convert card attributes to Chinese shorthand like '2实红菱'."""
    num = CHINESE_NUMBER.get(attrs['number'], attrs['number'])
    fill = CHINESE_FILL.get(attrs['fill'], attrs['fill'])
    color = CHINESE_COLOR.get(attrs['color'], attrs['color'])
    shape = CHINESE_SHAPE.get(attrs['shape'], attrs['shape'])
    return f"{num}{fill}{color}{shape}"

# Colors for highlighting Sets (RGB)
SET_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
]


class SetSolver:
    """End-to-end Set solver."""
    
    def __init__(
        self,
        detector_path: Optional[Path] = None,
        classifier_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        
        # Load detector
        if detector_path is None:
            detector_path = WEIGHTS_DIR / "detector" / "weights" / "best.pt"
        print(f"Loading detector from {detector_path}")
        self.detector = YOLO(str(detector_path))
        
        # Load classifier
        if classifier_path is None:
            classifier_path = WEIGHTS_DIR / "classifier_best.pt"
        print(f"Loading classifier from {classifier_path}")
        self.classifier = SetCardClassifier(pretrained=False)
        checkpoint = torch.load(classifier_path, map_location=device)
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.to(device)
        self.classifier.eval()
        
        # Classifier transform
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def detect_cards(self, image: Image.Image, conf: float = 0.5) -> List[dict]:
        """
        Detect cards in image.
        
        Returns list of detections with bounding boxes.
        """
        results = self.detector(image, conf=conf, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                })
        
        return detections
    
    def classify_card(self, card_image: Image.Image) -> dict:
        """Classify a cropped card image."""
        img_tensor = self.transform(card_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(img_tensor)
        
        result = {}
        for key, names in [
            ("number", NUMBER_NAMES),
            ("color", COLOR_NAMES),
            ("shape", SHAPE_NAMES),
            ("fill", FILL_NAMES),
        ]:
            probs = torch.softmax(outputs[key], dim=1)[0]
            pred_idx = probs.argmax().item()
            result[key] = names[pred_idx]
            result[f"{key}_conf"] = probs[pred_idx].item()
        
        return result
    
    def detection_to_card(self, attrs: dict, bbox: Tuple[int, int, int, int]) -> Card:
        """Convert classification result to Card object."""
        # Map classifier output to solver enums
        # Training data uses "blue" but standard Set calls it "purple"
        color_map = {"red": "RED", "green": "GREEN", "blue": "PURPLE"}
        # Training data uses "partial" for striped, "full" for solid
        fill_map = {"empty": "EMPTY", "full": "SOLID", "partial": "STRIPED"}
        
        return Card(
            shape=Shape[attrs["shape"].upper()],
            color=Color[color_map[attrs["color"]]],
            number=Number[attrs["number"].upper()],
            fill=Fill[fill_map[attrs["fill"]]],
            bbox=bbox,
        )
    
    def solve(
        self,
        image_path: str,
        conf: float = 0.5,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> dict:
        """
        Solve a Set game from image.
        
        Args:
            image_path: Path to input image
            conf: Detection confidence threshold
            output_path: Path to save annotated output image
            show: Whether to display the result
        
        Returns:
            Dict with detected cards and found Sets
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f"Loaded image: {image.size}")
        
        # Detect cards
        print("Detecting cards...")
        detections = self.detect_cards(image, conf=conf)
        print(f"Found {len(detections)} cards")
        
        # Classify each card
        print("Classifying cards...")
        cards = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            card_crop = image.crop((x1, y1, x2, y2))
            attrs = self.classify_card(card_crop)
            card = self.detection_to_card(attrs, det["bbox"])
            cards.append({
                "card": card,
                "attrs": attrs,
                "detection": det,
            })
        
        # Find Sets
        print("Finding Sets...")
        card_objects = [c["card"] for c in cards]
        sets = find_all_sets(card_objects)
        print(f"Found {len(sets)} valid Set(s)")
        
        # Draw results
        result_image = self._draw_results(image, cards, sets)
        
        if output_path:
            result_image.save(output_path)
            print(f"Saved result to {output_path}")
        
        if show:
            result_image.show()
        
        return {
            "num_cards": len(cards),
            "cards": [
                {
                    "attrs": c["attrs"],
                    "chinese": card_to_chinese(c["attrs"]),
                    "bbox": c["detection"]["bbox"],
                    "confidence": c["detection"]["confidence"],
                }
                for c in cards
            ],
            "num_sets": len(sets),
            "sets": [
                [str(card) for card in s]
                for s in sets
            ],
            "sets_chinese": [
                [card_to_chinese(next(c["attrs"] for c in cards if c["card"] is card)) for card in s]
                for s in sets
            ],
            "result_image": result_image,
        }
    
    def _draw_results(
        self,
        image: Image.Image,
        cards: List[dict],
        sets: List[Tuple[Card, Card, Card]],
    ) -> Image.Image:
        """Draw bounding boxes and Set highlights on image."""
        result = image.copy()
        draw = ImageDraw.Draw(result)
        
        # Try to load a Chinese-compatible font
        font = None
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "/System/Library/Fonts/STHeiti Light.ttc",  # macOS
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
            "C:\\Windows\\Fonts\\msyh.ttc",  # Windows
        ]
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 18)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
        
        # Map cards to Set colors (a card can be in multiple Sets)
        card_colors = {}  # card -> list of colors
        for i, card_set in enumerate(sets):
            color = SET_COLORS[i % len(SET_COLORS)]
            for card in card_set:
                if id(card) not in card_colors:
                    card_colors[id(card)] = []
                card_colors[id(card)].append(color)
        
        # Draw all cards
        for c in cards:
            card = c["card"]
            attrs = c["attrs"]
            x1, y1, x2, y2 = card.bbox
            
            # Determine box color
            if id(card) in card_colors:
                colors = card_colors[id(card)]
                color = colors[0]  # Primary color
                width = 4
            else:
                color = (200, 200, 200)  # Gray for non-Set cards
                width = 2
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            
            # Draw label (Chinese shorthand)
            label = card_to_chinese(attrs)
            draw.text((x1, y1 - 20), label, fill=color, font=font)
        
        # Draw Set count
        draw.text((10, 10), f"Found {len(sets)} Set(s)", fill=(255, 255, 255), font=font)
        
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Solve Set game from image")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, help="Path to save output image")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--show", action="store_true", help="Display result")
    args = parser.parse_args()
    
    solver = SetSolver()
    result = solver.solve(
        args.image,
        conf=args.conf,
        output_path=args.output,
        show=args.show,
    )
    
    print("\n" + "="*50)
    print("结果 RESULTS")
    print("="*50)
    print(f"检测到卡牌: {result['num_cards']}")
    print(f"找到Set: {result['num_sets']}")
    
    if result['cards']:
        print("\n卡牌:")
        for c in result['cards']:
            print(f"  {c['chinese']}")
    
    if result['sets_chinese']:
        print("\nSets:")
        for i, s in enumerate(result['sets_chinese'], 1):
            print(f"  Set {i}: {' + '.join(s)}")


if __name__ == "__main__":
    main()
