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

### ✅ Existing Data (FOUND!)
We have **1,943 labeled card images** in `training_images/`:
```
training_images/{number}/{color}/{shape}/{fill}/*.png
  - number: one, two, three
  - color: red, green, blue (blue = purple in standard Set)
  - shape: diamond, oval, squiggle
  - fill: empty, full, partial (partial = striped)
```
~20-30 images per class (81 classes). Perfect for classifier training!

### Synthetic Board Generation (For Detection)
Use existing card images to generate synthetic "board" images:

1. **Compose boards:** Place 12-15 cards on background images
2. **Augmentation Pipeline:**
   - Random orientations (0-360°)
   - Random positions (non-overlapping)
   - Background variations (tables, floors)
   - Lighting/color jitter
   - Perspective transforms
   - Scale variations

**Output:** YOLO-format annotations with bounding boxes

### Prior Art
- [tomwhite/set-game](https://github.com/tomwhite/set-game) - CNN approach with real photos
- [jgrodziski/set-game](https://github.com/jgrodziski/set-game) - Card generation logic

---

## V1 Milestones

### M1: Classification Model (Day 1-3) ⭐ START HERE
- [ ] Split existing data into train/val/test
- [ ] Train CNN classifier (MobileNetV3 for iPhone compat)
- [ ] Multi-head output: shape, color, number, fill
- [ ] Target: >95% accuracy per attribute
- [x] Set-finding algorithm (already implemented!)

### M2: Synthetic Board Generation (Day 4-6)
- [ ] Collect background images (tables, floors)
- [ ] Write board composition script
- [ ] Generate 5k+ board images with YOLO annotations
- [ ] Include rotation, scale, perspective augmentation

### M3: Detection Model (Day 7-9)
- [ ] Train YOLOv8n (nano) for card detection
- [ ] Evaluate on held-out synthetic + real photos
- [ ] Iterate on augmentation if needed

### M4: Integration Pipeline (Day 10-11)
- [ ] End-to-end: photo → detect → classify → solve
- [ ] Visualization: draw boxes + highlight Sets
- [ ] CLI tool: `python -m src.inference.solve photo.jpg`

### M5: Real-World Testing (Day 12-14)
- [ ] Test on real photos (take ~50 photos)
- [ ] Fine-tune if needed
- [ ] Performance benchmarks (speed, accuracy)

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
