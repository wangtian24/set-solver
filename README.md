# Set Solver 🃏

Vision-based solver for the [Set card game](https://www.setgame.com/).

Take a photo of Set cards → Get all valid Sets highlighted.

## Status

🚧 **V1 In Development**

## Features

- [x] Project setup
- [x] Set-finding algorithm
- [x] Card classifier (MobileNetV3)
- [x] Synthetic board generator
- [x] Card detector (YOLOv8)
- [x] End-to-end pipeline
- [ ] iPhone app (V2)

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training Pipeline

### 1. Train Classifier (on existing card images)
```bash
python -m src.train.classifier
# → saves weights/classifier_best.pt
```

### 2. Generate Synthetic Boards
```bash
python -m src.generate.board_generator --num 5000
# → creates data/synthetic/ with YOLO-format images + labels
```

### 3. Train Detector
```bash
python -m src.train.detector --epochs 100
# → saves weights/detector/weights/best.pt
```

### 4. Run Solver
```bash
python -m src.inference.solve photo.jpg -o result.jpg --show
```

## How It Works

1. **Detect** cards in the image using YOLOv8
2. **Classify** each card's 4 attributes (shape, color, number, fill)
3. **Solve** by finding all valid Sets (combinations where each attribute is all-same or all-different)
4. **Visualize** results with highlighted bounding boxes

## Set Card Properties

| Attribute | Values |
|-----------|--------|
| Shape | Diamond, Oval, Squiggle |
| Color | Red, Green, Purple |
| Number | 1, 2, 3 |
| Fill | Solid, Striped, Empty |

Total: 3^4 = **81 unique cards**

## Docs

- [Design Doc](docs/DESIGN.md) - Architecture and roadmap

## License

MIT
