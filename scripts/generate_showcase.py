#!/usr/bin/env python3
"""
Generate a showcase HTML page demonstrating Set solver on synthetic boards.

Creates an HTML page with:
- Sample synthetic board images with annotated results
- Per-board detection & classification accuracy vs ground truth
- Global accuracy summary
"""

import sys
import json
import random
import base64
from pathlib import Path
from io import BytesIO
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

# Reverse maps: index → name (must match classifier training)
NUMBER_NAMES = ["one", "two", "three"]
COLOR_NAMES = ["red", "green", "blue"]
SHAPE_NAMES = ["diamond", "oval", "squiggle"]
FILL_NAMES = ["empty", "full", "partial"]


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URI."""
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def iou(box_a, box_b):
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def gt_attrs_to_names(label: dict) -> dict:
    """Convert ground truth integer labels to name strings."""
    return {
        "number": NUMBER_NAMES[label["number"]],
        "color": COLOR_NAMES[label["color"]],
        "shape": SHAPE_NAMES[label["shape"]],
        "fill": FILL_NAMES[label["fill"]],
    }


def match_predictions_to_gt(predictions, gt_annotations, img_width, img_height):
    """
    Match predicted cards to ground truth annotations by IoU.

    Returns list of (pred, gt_names | None) tuples, plus unmatched GT count.
    """
    # Convert GT normalized coords to pixel bboxes
    gt_boxes = []
    for ann in gt_annotations:
        cx = ann["x_center"] * img_width
        cy = ann["y_center"] * img_height
        w = ann["width"] * img_width
        h = ann["height"] * img_height
        gt_boxes.append({
            "bbox": (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
            "attrs": gt_attrs_to_names(ann["label"]),
        })

    matched_gt = set()
    results = []

    for pred in predictions:
        best_iou = 0
        best_gt_idx = None
        pred_bbox = tuple(pred["bbox"])
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            score = iou(pred_bbox, gt["bbox"])
            if score > best_iou:
                best_iou = score
                best_gt_idx = j

        if best_gt_idx is not None and best_iou >= 0.3:
            matched_gt.add(best_gt_idx)
            results.append((pred, gt_boxes[best_gt_idx]["attrs"]))
        else:
            results.append((pred, None))  # false positive

    unmatched_gt = len(gt_boxes) - len(matched_gt)
    return results, unmatched_gt


def evaluate_board(predictions, gt_annotations, img_width, img_height):
    """
    Evaluate detection + classification accuracy for one board.

    Returns a dict with per-attribute and overall stats.
    """
    matches, missed_gt = match_predictions_to_gt(
        predictions, gt_annotations, img_width, img_height
    )

    gt_total = len(gt_annotations)
    detected = sum(1 for _, gt in matches if gt is not None)
    false_pos = sum(1 for _, gt in matches if gt is None)

    attr_correct = {"number": 0, "color": 0, "shape": 0, "fill": 0}
    attr_total = 0
    all_correct = 0  # cards where ALL 4 attributes match
    card_details = []  # per-card breakdown

    for pred, gt_attrs in matches:
        if gt_attrs is None:
            card_details.append({
                "pred": pred["chinese"],
                "gt": "???",
                "status": "false_pos",
                "wrong_attrs": [],
            })
            continue

        attr_total += 1
        wrong = []
        for attr in ["number", "color", "shape", "fill"]:
            if pred["attrs"][attr] == gt_attrs[attr]:
                attr_correct[attr] += 1
            else:
                wrong.append(attr)

        gt_chinese = _attrs_to_chinese(gt_attrs)
        if not wrong:
            all_correct += 1
            card_details.append({
                "pred": pred["chinese"],
                "gt": gt_chinese,
                "status": "correct",
                "wrong_attrs": [],
            })
        else:
            card_details.append({
                "pred": pred["chinese"],
                "gt": gt_chinese,
                "status": "wrong",
                "wrong_attrs": wrong,
            })

    return {
        "gt_total": gt_total,
        "detected": detected,
        "missed": missed_gt,
        "false_pos": false_pos,
        "attr_total": attr_total,
        "attr_correct": attr_correct,
        "all_correct": all_correct,
        "card_details": card_details,
    }


# Chinese shorthand helpers (duplicated here to keep script self-contained)
_CN_NUM = {"one": "1", "two": "2", "three": "3"}
_CN_FILL = {"full": "实", "empty": "空", "partial": "线"}
_CN_COLOR = {"red": "红", "green": "绿", "blue": "紫"}
_CN_SHAPE = {"diamond": "菱", "oval": "圆", "squiggle": "弯"}


def _attrs_to_chinese(attrs: dict) -> str:
    return (
        _CN_NUM.get(attrs["number"], "?")
        + _CN_FILL.get(attrs["fill"], "?")
        + _CN_COLOR.get(attrs["color"], "?")
        + _CN_SHAPE.get(attrs["shape"], "?")
    )


def generate_showcase(
    num_samples: int = 20,
    output_path: str = "showcase.html",
):
    """Generate showcase HTML page with accuracy evaluation."""

    from src.inference.solve import SetSolver

    print("Loading models...")
    solver = SetSolver()

    data_dir = Path.home() / "data" / "set-solver" / "synthetic"
    synthetic_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    # Support both flat and train/val directory structures
    all_images = sorted(synthetic_dir.rglob("*.jpg")) + sorted(synthetic_dir.rglob("*.png"))

    if not all_images:
        print("No synthetic images found!")
        return

    samples = random.sample(all_images, min(num_samples, len(all_images)))

    results = []
    # Global accumulators
    g_gt = g_det = g_missed = g_fp = 0
    g_attr_total = 0
    g_attr_correct = {"number": 0, "color": 0, "shape": 0, "fill": 0}
    g_all_correct = 0

    print(f"\nProcessing {len(samples)} images...")

    for i, img_path in enumerate(samples):
        stem = img_path.stem
        # Support train/val subdirectory structure
        rel = img_path.relative_to(synthetic_dir)
        json_path = labels_dir / rel.parent / f"{stem}.json"
        print(f"  [{i+1}/{len(samples)}] {img_path.name}...", end=" ")

        try:
            result = solver.solve(str(img_path), conf=0.25)
            annotated = result["result_image"]
            img = Image.open(img_path)
            img_w, img_h = img.size

            # Load ground truth
            if json_path.exists():
                with open(json_path) as f:
                    gt_annotations = json.load(f)
            else:
                gt_annotations = []

            eval_info = evaluate_board(
                result["cards"], gt_annotations, img_w, img_h
            )

            # Accumulate globals
            g_gt += eval_info["gt_total"]
            g_det += eval_info["detected"]
            g_missed += eval_info["missed"]
            g_fp += eval_info["false_pos"]
            g_attr_total += eval_info["attr_total"]
            g_all_correct += eval_info["all_correct"]
            for k in g_attr_correct:
                g_attr_correct[k] += eval_info["attr_correct"][k]

            det_acc = eval_info["detected"] / eval_info["gt_total"] * 100 if eval_info["gt_total"] else 0
            cls_acc = eval_info["all_correct"] / eval_info["attr_total"] * 100 if eval_info["attr_total"] else 0
            print(f"det {eval_info['detected']}/{eval_info['gt_total']} "
                  f"cls {eval_info['all_correct']}/{eval_info['attr_total']}")

            results.append({
                "name": img_path.name,
                "annotated_b64": image_to_base64(annotated),
                "num_cards": result["num_cards"],
                "num_sets": result["num_sets"],
                "sets": result["sets"],
                "sets_chinese": result.get("sets_chinese", []),
                "cards": result["cards"],
                "eval": eval_info,
            })

        except Exception as e:
            print(f"Error: {e}")
            continue

    if not results:
        print("No results to display!")
        return

    # Global stats
    g_det_pct = g_det / g_gt * 100 if g_gt else 0
    g_cls_pct = g_all_correct / g_attr_total * 100 if g_attr_total else 0
    g_attr_pcts = {}
    for k in ["number", "color", "shape", "fill"]:
        g_attr_pcts[k] = g_attr_correct[k] / g_attr_total * 100 if g_attr_total else 0

    print(f"\n{'='*50}")
    print(f"Detection: {g_det}/{g_gt} ({g_det_pct:.1f}%)  Missed: {g_missed}  FP: {g_fp}")
    print(f"Classification (all-4-correct): {g_all_correct}/{g_attr_total} ({g_cls_pct:.1f}%)")
    for k in ["number", "color", "shape", "fill"]:
        print(f"  {k}: {g_attr_correct[k]}/{g_attr_total} ({g_attr_pcts[k]:.1f}%)")

    # ========== Generate HTML ==========
    print("\nGenerating HTML...")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Set Solver Showcase</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}

        header {{ text-align: center; margin-bottom: 50px; }}
        h1 {{
            font-size: 3rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #f39c12, #e74c3c, #9b59b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ color: #8e9aaf; font-size: 1.2rem; }}

        .stats {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 20px;
            margin: 30px 0;
        }}
        .stat {{
            text-align: center;
            padding: 20px 30px;
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            min-width: 140px;
        }}
        .stat-value {{ font-size: 2.2rem; font-weight: bold; color: #f39c12; }}
        .stat-label {{ color: #8e9aaf; margin-top: 5px; font-size: 0.9rem; }}

        .attr-stats {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 12px;
            margin: 10px 0 40px 0;
        }}
        .attr-stat {{
            background: rgba(255,255,255,0.07);
            border-radius: 8px;
            padding: 10px 18px;
            text-align: center;
            font-size: 0.85rem;
        }}
        .attr-stat .val {{ font-size: 1.3rem; font-weight: bold; }}
        .attr-stat .val.perfect {{ color: #2ecc71; }}
        .attr-stat .val.good {{ color: #f39c12; }}
        .attr-stat .val.bad {{ color: #e74c3c; }}
        .attr-stat .lbl {{ color: #8e9aaf; }}

        .grid {{
            display: flex;
            flex-direction: column;
            gap: 30px;
        }}

        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
            display: grid;
            grid-template-columns: 2fr 1fr;
        }}
        .card-header {{
            grid-column: 1 / -1;
            padding: 16px 20px;
            background: rgba(0,0,0,0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .card-title {{ font-weight: 600; }}
        .badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            white-space: nowrap;
        }}
        .badge.green {{ background: #27ae60; }}
        .badge.red {{ background: #e74c3c; }}
        .badge.gray {{ background: #555; }}
        .badge.yellow {{ background: #d4a017; color: #000; }}

        .image-container img {{
            width: 100%;
            height: auto;
            display: block;
        }}

        .info-panel {{
            padding: 16px;
            background: rgba(0,0,0,0.1);
            overflow-y: auto;
            font-size: 0.85rem;
            line-height: 1.7;
        }}
        .info-panel h4 {{ margin-bottom: 8px; color: #f39c12; }}
        .card-row {{ padding: 3px 0; }}
        .card-row.correct {{ color: #2ecc71; }}
        .card-row.wrong {{ color: #e74c3c; }}
        .card-row.false_pos {{ color: #888; text-decoration: line-through; }}
        .set-item {{ padding: 4px 0; color: #8e9aaf; }}

        footer {{
            text-align: center;
            margin-top: 50px;
            color: #8e9aaf;
        }}

        @media (max-width: 700px) {{
            .card {{ grid-template-columns: 1fr; }}
            .stats {{ flex-direction: column; gap: 15px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Set Solver Showcase</h1>
            <p class="subtitle">Detection + Classification accuracy on synthetic boards</p>
        </header>

        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(results)}</div>
                <div class="stat-label">Boards</div>
            </div>
            <div class="stat">
                <div class="stat-value">{g_det}/{g_gt}</div>
                <div class="stat-label">Detection ({g_det_pct:.1f}%)</div>
            </div>
            <div class="stat">
                <div class="stat-value">{g_all_correct}/{g_attr_total}</div>
                <div class="stat-label">Classification ({g_cls_pct:.1f}%)</div>
            </div>
            <div class="stat">
                <div class="stat-value">{g_missed}</div>
                <div class="stat-label">Missed Cards</div>
            </div>
            <div class="stat">
                <div class="stat-value">{g_fp}</div>
                <div class="stat-label">False Positives</div>
            </div>
        </div>

        <div class="attr-stats">"""

    for k in ["number", "color", "shape", "fill"]:
        pct = g_attr_pcts[k]
        cls = "perfect" if pct >= 99.5 else "good" if pct >= 95 else "bad"
        html += f"""
            <div class="attr-stat">
                <div class="val {cls}">{pct:.1f}%</div>
                <div class="lbl">{k}</div>
            </div>"""

    html += """
        </div>

        <div class="grid">
"""

    for r in results:
        ev = r["eval"]
        det_pct = ev["detected"] / ev["gt_total"] * 100 if ev["gt_total"] else 0
        cls_pct = ev["all_correct"] / ev["attr_total"] * 100 if ev["attr_total"] else 0

        det_cls = "green" if det_pct >= 99.5 else "yellow" if det_pct >= 80 else "red"
        cls_cls = "green" if cls_pct >= 99.5 else "yellow" if cls_pct >= 80 else "red"
        set_cls = "green" if r["num_sets"] > 0 else "gray"

        # Build per-card detail rows
        detail_html = '<div class="info-panel">'
        detail_html += f'<h4>Detection {ev["detected"]}/{ev["gt_total"]} | Classification {ev["all_correct"]}/{ev["attr_total"]}</h4>'

        for cd in ev["card_details"]:
            if cd["status"] == "correct":
                detail_html += f'<div class="card-row correct">{cd["pred"]} = {cd["gt"]}</div>'
            elif cd["status"] == "wrong":
                wrong_str = ",".join(cd["wrong_attrs"])
                detail_html += f'<div class="card-row wrong">{cd["pred"]} != {cd["gt"]} [{wrong_str}]</div>'
            else:
                detail_html += f'<div class="card-row false_pos">{cd["pred"]} (no match)</div>'

        if ev["missed"] > 0:
            detail_html += f'<div class="card-row wrong">{ev["missed"]} card(s) missed by detector</div>'

        if r["sets_chinese"]:
            detail_html += "<h4 style='margin-top:12px'>Sets</h4>"
            for si, s in enumerate(r["sets_chinese"], 1):
                detail_html += f'<div class="set-item">Set {si}: {" + ".join(s)}</div>'

        detail_html += "</div>"

        html += f"""
            <div class="card">
                <div class="card-header">
                    <span class="card-title">{r['name']}</span>
                    <span>
                        <span class="badge {det_cls}">Det {ev['detected']}/{ev['gt_total']}</span>
                        <span class="badge {cls_cls}">Cls {ev['all_correct']}/{ev['attr_total']}</span>
                        <span class="badge {set_cls}">{r['num_sets']} set(s)</span>
                    </span>
                </div>
                <div class="image-container">
                    <img src="{r['annotated_b64']}" alt="Detected">
                </div>
                {detail_html}
            </div>
"""

    html += f"""
        </div>

        <footer>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            <p>Set Solver v1.0</p>
        </footer>
    </div>
</body>
</html>
"""

    output = PROJECT_ROOT / output_path
    output.write_text(html)
    print(f"\nShowcase saved to: {output}")
    print(f"Open in browser: file://{output.absolute()}")
    return str(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Set solver showcase")
    parser.add_argument("--samples", "-n", type=int, default=20, help="Number of sample images")
    parser.add_argument("--output", "-o", type=str, default="showcase.html", help="Output path")
    args = parser.parse_args()

    generate_showcase(num_samples=args.samples, output_path=args.output)
