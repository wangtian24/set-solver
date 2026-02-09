# Set Solver 🃏

Vision-based solver for the [Set card game](https://www.setgame.com/).

Take a photo of Set cards → Get all valid Sets highlighted.

## Status

🚧 **V1 In Development**

## Features (Planned)

- [x] Project setup
- [ ] Synthetic training data generation
- [ ] Card detection (YOLOv8)
- [ ] Card classification (CNN)
- [ ] Set-finding algorithm
- [ ] CLI tool
- [ ] iPhone app (V2)

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Solve a photo (coming soon)
python -m src.inference.solve photo.jpg
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
