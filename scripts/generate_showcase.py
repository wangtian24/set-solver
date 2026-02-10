#!/usr/bin/env python3
"""
Generate a showcase HTML page demonstrating Set solver on synthetic boards.

Creates an HTML page with:
- Sample synthetic board images
- Annotated versions showing detected cards
- Highlighted Sets
- Accuracy metrics
"""

import sys
import random
import base64
from pathlib import Path
from io import BytesIO
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URI."""
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def generate_showcase(
    num_samples: int = 6,
    output_path: str = "showcase.html",
):
    """Generate showcase HTML page."""
    
    from src.inference.solve import SetSolver
    
    # Initialize solver
    print("Loading models...")
    solver = SetSolver()
    
    # Get sample images
    synthetic_dir = PROJECT_ROOT / "data" / "synthetic" / "images"
    all_images = list(synthetic_dir.glob("*.jpg"))
    
    if not all_images:
        print("No synthetic images found!")
        return
    
    # Select random samples
    samples = random.sample(all_images, min(num_samples, len(all_images)))
    
    results = []
    total_cards = 0
    total_sets = 0
    
    print(f"\nProcessing {len(samples)} images...")
    
    for i, img_path in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {img_path.name}...")
        
        try:
            result = solver.solve(str(img_path))
            
            # Store results
            original = Image.open(img_path)
            annotated = result["result_image"]
            
            results.append({
                "name": img_path.name,
                "original_b64": image_to_base64(original),
                "annotated_b64": image_to_base64(annotated),
                "num_cards": result["num_cards"],
                "num_sets": result["num_sets"],
                "sets": result["sets"],
                "cards": result["cards"],
            })
            
            total_cards += result["num_cards"]
            total_sets += result["num_sets"]
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    if not results:
        print("No results to display!")
        return
    
    # Generate HTML
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
        
        header {{
            text-align: center;
            margin-bottom: 50px;
        }}
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
            gap: 40px;
            margin: 30px 0;
        }}
        .stat {{
            text-align: center;
            padding: 20px 40px;
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
        }}
        .stat-value {{ font-size: 2.5rem; font-weight: bold; color: #f39c12; }}
        .stat-label {{ color: #8e9aaf; margin-top: 5px; }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
        }}
        
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card-header {{
            padding: 20px;
            background: rgba(0,0,0,0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .card-title {{ font-weight: 600; }}
        .badge {{
            background: #e74c3c;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9rem;
        }}
        .badge.green {{ background: #27ae60; }}
        
        .images {{
            display: grid;
            grid-template-columns: 1fr 1fr;
        }}
        .image-container {{
            position: relative;
        }}
        .image-container img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .image-label {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.85rem;
        }}
        
        .sets-list {{
            padding: 20px;
            background: rgba(0,0,0,0.1);
        }}
        .sets-list h4 {{ margin-bottom: 10px; color: #f39c12; }}
        .set-item {{
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            font-size: 0.9rem;
            color: #8e9aaf;
        }}
        .set-item:last-child {{ border-bottom: none; }}
        
        footer {{
            text-align: center;
            margin-top: 50px;
            color: #8e9aaf;
        }}
        
        @media (max-width: 700px) {{
            .grid {{ grid-template-columns: 1fr; }}
            .images {{ grid-template-columns: 1fr; }}
            .stats {{ flex-direction: column; gap: 15px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🃏 Set Solver Showcase</h1>
            <p class="subtitle">Vision-based Set card game solver</p>
        </header>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(results)}</div>
                <div class="stat-label">Boards Analyzed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{total_cards}</div>
                <div class="stat-label">Cards Detected</div>
            </div>
            <div class="stat">
                <div class="stat-value">{total_sets}</div>
                <div class="stat-label">Sets Found</div>
            </div>
        </div>
        
        <div class="grid">
"""
    
    for r in results:
        sets_html = ""
        if r["sets"]:
            sets_html = '<div class="sets-list"><h4>Valid Sets:</h4>'
            for i, s in enumerate(r["sets"][:5], 1):  # Limit to 5
                sets_html += f'<div class="set-item">Set {i}: {", ".join(s)}</div>'
            if len(r["sets"]) > 5:
                sets_html += f'<div class="set-item">...and {len(r["sets"]) - 5} more</div>'
            sets_html += '</div>'
        
        badge_class = "green" if r["num_sets"] > 0 else ""
        
        html += f"""
            <div class="card">
                <div class="card-header">
                    <span class="card-title">{r['name']}</span>
                    <span class="badge {badge_class}">{r['num_cards']} cards / {r['num_sets']} sets</span>
                </div>
                <div class="images">
                    <div class="image-container">
                        <img src="{r['original_b64']}" alt="Original">
                        <span class="image-label">Original</span>
                    </div>
                    <div class="image-container">
                        <img src="{r['annotated_b64']}" alt="Detected">
                        <span class="image-label">Detected</span>
                    </div>
                </div>
                {sets_html}
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
    
    # Write HTML
    output = PROJECT_ROOT / output_path
    output.write_text(html)
    print(f"\n✅ Showcase saved to: {output}")
    print(f"   Open in browser: file://{output.absolute()}")
    
    return str(output)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Set solver showcase")
    parser.add_argument("--samples", "-n", type=int, default=6, help="Number of sample images")
    parser.add_argument("--output", "-o", type=str, default="showcase.html", help="Output path")
    args = parser.parse_args()
    
    generate_showcase(num_samples=args.samples, output_path=args.output)
