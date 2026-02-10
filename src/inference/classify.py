"""
Inference script for classifying a single card image.
"""

import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Import from training module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.train.classifier import (
    SetCardClassifier,
    NUMBER_NAMES, COLOR_NAMES, SHAPE_NAMES, FILL_NAMES
)

WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"


def load_model(weights_path: Path = None, device: str = None):
    """Load trained classifier."""
    if weights_path is None:
        weights_path = WEIGHTS_DIR / "classifier_best.pt"
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    model = SetCardClassifier(pretrained=False)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, device


def classify_card(image: Image.Image, model, device) -> dict:
    """
    Classify a card image.
    
    Returns dict with predicted attributes and confidences.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Get predictions and confidences
    result = {}
    for key, names in [
        ("number", NUMBER_NAMES),
        ("color", COLOR_NAMES),
        ("shape", SHAPE_NAMES),
        ("fill", FILL_NAMES),
    ]:
        probs = torch.softmax(outputs[key], dim=1)[0]
        pred_idx = probs.argmax().item()
        result[key] = {
            "value": names[pred_idx],
            "confidence": probs[pred_idx].item(),
            "all_probs": {name: probs[i].item() for i, name in enumerate(names)},
        }
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify a Set card image")
    parser.add_argument("image", type=str, help="Path to card image")
    args = parser.parse_args()
    
    print("Loading model...")
    model, device = load_model()
    
    print(f"Classifying {args.image}...")
    image = Image.open(args.image).convert("RGB")
    result = classify_card(image, model, device)
    
    print("\nPrediction:")
    print(f"  Number: {result['number']['value']} ({result['number']['confidence']:.1%})")
    print(f"  Color:  {result['color']['value']} ({result['color']['confidence']:.1%})")
    print(f"  Shape:  {result['shape']['value']} ({result['shape']['confidence']:.1%})")
    print(f"  Fill:   {result['fill']['value']} ({result['fill']['confidence']:.1%})")
    
    # Human-readable card name
    n = result['number']['value']
    c = result['color']['value']
    s = result['shape']['value']
    f = result['fill']['value']
    print(f"\nCard: {n} {f} {c} {s}(s)")


if __name__ == "__main__":
    main()
