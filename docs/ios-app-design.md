# iOS Set Solver App — Design Document

## Context

The Set Solver currently runs as a Python web app (FastAPI + YOLO + MobileNetV3). The goal is to ship it as a native iOS app running entirely on-device using CoreML, publishable to the App Store.

## Approach: Native SwiftUI + CoreML

Two phases: (1) export models to CoreML via Python, (2) build a SwiftUI app.

---

## Phase 1: CoreML Model Export

**New file: `scripts/export_coreml.py`**

### 1A. Detector (YOLO11n → CoreML)

```python
from ultralytics import YOLO
model = YOLO("weights/detector/weights/best.pt")
model.export(format="coreml", imgsz=640, nms=True)
```

Ultralytics handles everything. Output: `.mlpackage` (~5MB).

### 1B. Classifier (MobileNetV3 multi-head → CoreML)

The multi-head dict output doesn't trace cleanly. Solution: wrap it to concatenate the 4 heads into a single `(1, 12)` tensor, then trace + convert via `coremltools`.

```python
class ClassifierWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return torch.cat([out["number"], out["color"], out["shape"], out["fill"]], dim=1)

# Load, wrap, trace, convert
model = SetCardClassifier(pretrained=False)
checkpoint = torch.load("weights/classifier_best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

wrapper = ClassifierWrapper(model)
traced = torch.jit.trace(wrapper, torch.randn(1, 3, 224, 224))

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=(1, 3, 224, 224))],
    minimum_deployment_target=ct.target.iOS16,
)
mlmodel.save("SetCardClassifier.mlpackage")
```

- Output indices: `[0:3]` = number, `[3:6]` = color, `[6:9]` = shape, `[9:12]` = fill
- Use `ct.TensorType` input (not ImageType) — do all preprocessing in Swift for full control
- Requires: `pip install coremltools>=7.0`

**Source files:**
- `src/train/classifier.py` — `SetCardClassifier` class, attribute name lists (lines 30-39, 113-149)

---

## Phase 2: Xcode Project

### Structure

```
ios/SetSolver/
├── SetSolverApp.swift            # @main entry
├── ContentView.swift             # 3-state UI (idle/scanning/frozen)
├── CameraManager.swift           # AVFoundation capture
├── CameraPreview.swift           # UIViewRepresentable
├── SetSolverPipeline.swift       # CoreML inference orchestration
├── SetFinder.swift               # Set-finding algorithm (pure Swift)
├── Card.swift                    # Card struct + attribute enums
├── TrophyBar.swift               # Cropped card strip
├── ResultOverlayView.swift       # Bbox drawing on frozen frame
└── Resources/
    ├── SetCardDetector.mlpackage
    └── SetCardClassifier.mlpackage
```

~9 Swift files total.

### Key Components

**Card.swift** — Port enums/struct from `src/solver/set_finder.py`. Match enum raw values to classifier output order:
- Number: one=0, two=1, three=2
- Color: red=0, green=1, purple=2
- Shape: diamond=0, oval=1, squiggle=2
- Fill: empty=0, solid=1, striped=2

**SetFinder.swift** — Brute-force C(n,3) from `src/solver/set_finder.py:80-110`. For each attribute: `Set([v1,v2,v3]).count == 2` is invalid. ~20 lines of Swift.

**CameraManager.swift** — `AVCaptureSession` with `.hd1280x720`, back camera, `AVCaptureVideoDataOutput`. Delivers `CVPixelBuffer` frames via delegate.

**SetSolverPipeline.swift** — Replicates `src/inference/solve.py`:
1. Run YOLO via `VNCoreMLRequest` on `CVPixelBuffer` → bounding boxes
2. Filter merged detections (area > 2.2x median) — from `solve.py:121-124`
3. Crop each card → resize 224x224 → ImageNet normalize → run classifier
4. Parse `(1,12)` output → argmax each 3-slice → `Card`
5. Call `findAllSets()` → return results

Skip frames if still processing (matches web app's `processing` guard).

**ContentView.swift** — 3-state flow matching web app (`index.html`):
- **Idle**: camera preview + green "Start" button
- **Scanning**: camera preview + red "Stop" button + ~3fps inference
- **Frozen**: result image + yellow "Restart" button + trophy bar + left/right set nav

**TrophyBar.swift** — Horizontal row of 3 cropped card images (60px height, capped width).

**ResultOverlayView.swift** — Draw colored bboxes + Chinese labels on the captured frame. Only highlight the currently selected set. Chinese shorthand format: `2实红菱` (from `solve.py:29-42`).

### Info.plist
- `NSCameraUsageDescription`: camera permission string

### Deployment Target
- iOS 16.0 (modern CoreML + SwiftUI)

---

## Implementation Order

| Step | What | Testable? |
|------|------|-----------|
| 1 | `scripts/export_coreml.py` — export both models | Run script, verify `.mlpackage` files |
| 2 | Create Xcode project, add models | Build succeeds, Xcode generates model classes |
| 3 | `Card.swift` + `SetFinder.swift` | Unit test with known card combos |
| 4 | `CameraManager.swift` + `CameraPreview.swift` | Camera preview shows on device |
| 5 | `SetSolverPipeline.swift` | Test with a static photo first |
| 6 | `ContentView.swift` + overlays + trophy | Full end-to-end on device |

---

## Model Sizes & Performance

- Detector: ~5MB, ~30ms on Neural Engine
- Classifier: ~11MB, ~5ms per card x 12 cards = ~60ms
- Total per frame: ~100ms → 3fps easily achievable on iPhone 8+
- App bundle: ~20MB total

## Risks

| Risk | Mitigation |
|------|------------|
| YOLO `nms=True` export incompatible with Vision framework | Fall back to `nms=False`, implement NMS in Swift |
| CoreML preprocessing mismatch | Use TensorType + manual preprocessing in Swift |
| Older iPhones too slow | iPhone 8+ has Neural Engine; show processing indicator |

## Key Reference Files

| File | What to port |
|------|-------------|
| `src/solver/set_finder.py:38-110` | Card model + set-finding algorithm |
| `src/inference/solve.py:97-163` | Detection, classification, attribute mapping |
| `src/inference/solve.py:29-42` | Chinese shorthand labels |
| `src/train/classifier.py:30-39` | Attribute name/index mapping |
| `src/web/templates/index.html` | UI flow (3-state, trophy, nav arrows) |
