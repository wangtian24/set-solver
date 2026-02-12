"""
Web-based real-time Set solver.

FastAPI backend serving a single HTML page with live camera feed.
Processes frames via the SetSolver pipeline and returns annotated results.
"""

import base64
import io
import sys
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.solve import SetSolver

app = FastAPI(title="Set Solver")

# Global solver instance (loaded once at startup)
solver: SetSolver = None


@app.on_event("startup")
def load_solver():
    global solver
    print("Loading Set Solver pipeline...")
    solver = SetSolver()
    print("Solver ready!")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return html_path.read_text()


@app.post("/api/solve")
async def solve_frame(file: UploadFile = File(...)):
    """Accept a JPEG frame, run solver, return results."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = solver.solve_from_image(image, conf=0.7, cls_conf=0.8)

    # Encode per-set annotated images as base64 JPEG
    result_images_b64 = []
    for img in result.pop("result_images"):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        result_images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    result["result_images_b64"] = result_images_b64

    # Crop cards per set for trophy display
    per_set_cards_b64 = []
    for bboxes in result.get("sets_bboxes", []):
        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            crop = image.crop((x1, y1, x2, y2))
            cbuf = io.BytesIO()
            crop.save(cbuf, format="JPEG", quality=90)
            crops.append(base64.b64encode(cbuf.getvalue()).decode("utf-8"))
        per_set_cards_b64.append(crops)
    result["per_set_cards_b64"] = per_set_cards_b64

    return result


if __name__ == "__main__":
    import argparse
    import subprocess
    import tempfile
    import uvicorn

    parser = argparse.ArgumentParser(description="Set Solver web server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-ssl", action="store_true", help="Disable auto-generated SSL (camera requires HTTPS on non-localhost)")
    args = parser.parse_args()

    ssl_kwargs = {}
    if not args.no_ssl:
        # Generate a self-signed cert so mobile browsers allow camera access
        cert_dir = Path(tempfile.mkdtemp())
        cert_file = cert_dir / "cert.pem"
        key_file = cert_dir / "key.pem"
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key_file), "-out", str(cert_file),
            "-days", "1", "-nodes",
            "-subj", "/CN=set-solver",
        ], check=True, capture_output=True)
        ssl_kwargs = {"ssl_certfile": str(cert_file), "ssl_keyfile": str(key_file)}
        proto = "https"
    else:
        proto = "http"

    # Show access URLs
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "127.0.0.1"
    print(f"\n  Set Solver running at:")
    print(f"    Local:   {proto}://localhost:{args.port}")
    print(f"    Network: {proto}://{local_ip}:{args.port}\n")

    uvicorn.run("src.web.app:app", host="0.0.0.0", port=args.port, reload=False, **ssl_kwargs)
