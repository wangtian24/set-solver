# Set Solver - Design Doc

## Overview

A vision-based Set card game solver that:
1. Takes a photo of Set cards laid out on a table
2. Detects and classifies all cards (any orientation)
3. Finds all valid Sets
4. Marks the solutions on the image

**End goal:** iPhone app for real-time Set solving.

---

## Set Card Properties

Each card has 4 attributes with 3 values each (3^4 = 81 unique cards):

| Attribute | Values |
|-----------|--------|
| **Shape** | Diamond, Oval, Squiggle |
| **Color** | Red, Green, Purple |
| **Number** | 1, 2, 3 |
| **Fill** | Solid, Striped, Empty |

A **valid Set** = 3 cards where each attribute is either ALL SAME or ALL DIFFERENT.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Input      │ ──▶ │  Detection   │ ──▶ │  Classify   │ ──▶ │  Set Solver  │
│  Photo      │     │  (YOLOv8)    │     │  (CNN)      │     │  Algorithm   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                           │                    │                    │
                           ▼                    ▼                    ▼
                      Bounding boxes      Card attributes       Valid Sets
                      + orientations      (81 classes)          + positions
```

### Option A: Two-Stage Pipeline (Recommended for v1)
1. **YOLOv8** for card detection + orientation
2. **CNN classifier** for card attributes (81 classes or 4 multi-label heads)

### Option B: Single-Stage
- YOLOv8 with 81 classes (one per card type)
- Simpler but needs more training data per class

**Decision:** Start with Option A - more modular, easier to debug.

---

## Training Data Strategy

### Synthetic Generation (Primary - FREE)
Generate training images programmatically:

1. **Card SVGs/PNGs:** Create clean vector art for all 81 cards
2. **Augmentation Pipeline:**
   - Random backgrounds (tables, floors, etc.)
   - Random orientations (0-360°)
   - Lighting variations
   - Perspective transforms
   - Partial occlusions
   - Camera blur/noise
   - Scale variations

**Tools:** Python + PIL/OpenCV + Albumentations

### Real Photos (Validation)
- Take ~100-200 real photos for validation set
- Use existing datasets (tomwhite/set-game has labeled data)
- Crowdsource from friends/family playing Set

### Prior Art
- [tomwhite/set-game](https://github.com/tomwhite/set-game) - CNN approach with real photos
- [jgrodziski/set-game](https://github.com/jgrodziski/set-game) - Card generation logic

---

## V1 Milestones

### M1: Card Asset Creation (Day 1-2)
- [ ] Generate SVG/PNG for all 81 cards
- [ ] Create card outline for detection training
- [ ] Validate visual accuracy

### M2: Synthetic Data Pipeline (Day 3-5)
- [ ] Background image collection (textures, tables)
- [ ] Data generation script
- [ ] Generate 10k+ training images with labels
- [ ] YOLO format annotations (bounding boxes)
- [ ] Classification labels

### M3: Detection Model (Day 6-8)
- [ ] Train YOLOv8 for card detection
- [ ] Handle rotated bounding boxes (OBB)
- [ ] Evaluate on real photos
- [ ] Iterate on data augmentation

### M4: Classification Model (Day 9-11)
- [ ] Train CNN classifier (ResNet18 or MobileNet)
- [ ] Multi-head output: shape, color, number, fill
- [ ] Evaluate accuracy per attribute

### M5: Solver + Visualization (Day 12-13)
- [ ] Implement Set-finding algorithm (brute force: C(n,3) combinations)
- [ ] Draw bounding boxes + labels on image
- [ ] Highlight valid Sets with colors

### M6: Integration + CLI (Day 14)
- [ ] End-to-end pipeline: photo → solutions
- [ ] CLI tool: `set-solver solve image.jpg`
- [ ] Performance benchmarks

---

## iPhone Deployment (V2)

### CoreML Conversion
```
PyTorch → ONNX → CoreML
or
PyTorch → coremltools directly
```

### App Architecture
```
┌─────────────────────────────────────────────┐
│  SwiftUI App                                │
│  ┌──────────────────────────────────────┐   │
│  │  Camera View                         │   │
│  │  - Real-time preview                 │   │
│  │  - Tap to capture                    │   │
│  └──────────────────────────────────────┘   │
│  ┌──────────────────────────────────────┐   │
│  │  CoreML Inference                    │   │
│  │  - Detection model                   │   │
│  │  - Classification model              │   │
│  └──────────────────────────────────────┘   │
│  ┌──────────────────────────────────────┐   │
│  │  Results Overlay                     │   │
│  │  - Bounding boxes                    │   │
│  │  - Set highlights                    │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### Considerations
- Model size: Target < 10MB for fast loading
- Use MobileNetV3 or EfficientNet-Lite for classification
- YOLOv8n (nano) for detection
- Quantization for speed

---

## File Structure

```
set-solver/
├── README.md
├── docs/
│   └── DESIGN.md
├── assets/
│   └── cards/           # SVG/PNG for all 81 cards
├── data/
│   ├── synthetic/       # Generated training data
│   ├── real/            # Real photos for validation
│   └── backgrounds/     # Table textures, etc.
├── src/
│   ├── generate/        # Synthetic data generation
│   ├── train/           # Training scripts
│   ├── models/          # Model definitions
│   ├── solver/          # Set-finding algorithm
│   └── inference/       # Inference pipeline
├── notebooks/           # Experiments
├── weights/             # Trained model weights
└── ios/                 # Future iPhone app
```

---

## Dependencies

```
# Core
python >= 3.10
pytorch >= 2.0
ultralytics (YOLOv8)
albumentations
opencv-python
pillow

# Optional
coremltools          # iOS export
svglib               # SVG handling
cairosvg             # SVG → PNG
```

---

## Open Questions

1. **Rotated bounding boxes vs axis-aligned?**
   - YOLOv8-OBB supports oriented boxes
   - Might be overkill if we normalize orientation in preprocessing

2. **81 classes vs multi-head?**
   - Multi-head (4 outputs) needs less data per combination
   - 81 classes is simpler conceptually

3. **Real-time on iPhone feasible?**
   - Need to benchmark: target 15+ FPS
   - May need to process every Nth frame

---

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [tomwhite/set-game](https://github.com/tomwhite/set-game)
- [Set Game Rules](https://www.setgame.com/set/puzzle)
- [CoreML Tools](https://coremltools.readme.io/)
