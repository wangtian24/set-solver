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

    result = solver.solve_from_image(image, conf=0.25)

    # Encode annotated image as base64 JPEG
    result_image = result.pop("result_image")
    buf = io.BytesIO()
    result_image.save(buf, format="JPEG", quality=85)
    result["result_image_b64"] = base64.b64encode(buf.getvalue()).decode("utf-8")

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000, reload=False)
