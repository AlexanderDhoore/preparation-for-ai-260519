from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, "/root/preparation-for-ai/08-practical-mlops")

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from mlops_lab_common import build_mobilenet, read_json

LAB_DIR = Path("/root/preparation-for-ai/08-practical-mlops/lab2")
BUNDLE_DIR = Path("/root/preparation-for-ai/08-practical-mlops/lab1/model_bundle")

app = FastAPI(
    title="Preparation For AI Image Model API",
    description="Small FastAPI service for the Chapter 8 model bundle.",
)


def load_model_bundle():
    if not BUNDLE_DIR.exists():
        raise RuntimeError(
            "Model bundle not found. Run "
            "08-practical-mlops/lab1/track-image-model.py first."
        )

    labels = read_json(BUNDLE_DIR / "labels.json")
    config = read_json(BUNDLE_DIR / "model_config.json")
    model, transform = build_mobilenet(
        output_classes=len(labels),
        trainable_backbone_blocks=int(config["trainable_backbone_blocks"]),
    )
    state = torch.load(BUNDLE_DIR / "model_state.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return labels, config, model, transform


LABELS, CONFIG, MODEL, TRANSFORM = load_model_bundle()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    return {
        "model_name": CONFIG["model_name"],
        "label_count": len(LABELS),
        "labels": LABELS,
        "manifest_key": CONFIG["manifest_key"],
        "mlflow_run_id": CONFIG["mlflow_run_id"],
        "mlflow_run_name": CONFIG["mlflow_run_name"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    with Image.open(BytesIO(content)) as image:
        tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, prediction_index = probabilities.max(dim=0)

    return {
        "filename": file.filename,
        "predicted_label": LABELS[int(prediction_index)],
        "confidence": round(float(confidence), 4),
        "model_name": CONFIG["model_name"],
        "mlflow_run_id": CONFIG["mlflow_run_id"],
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html>
      <head>
        <title>Preparation For AI Model API</title>
        <style>
          body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto; }
          img { max-width: 260px; display: block; margin-top: 16px; }
          pre { background: #f4f4f4; padding: 12px; white-space: pre-wrap; }
        </style>
      </head>
      <body>
        <h1>Image Model API</h1>
        <p>Upload an image and call the FastAPI prediction endpoint.</p>
        <input id="file" type="file" accept="image/*" />
        <button onclick="predict()">Predict</button>
        <img id="preview" />
        <pre id="result"></pre>
        <p><a href="/docs">Open API docs</a></p>
        <script>
          async function predict() {
            const file = document.getElementById("file").files[0];
            if (!file) {
              document.getElementById("result").textContent = "Choose an image first.";
              return;
            }
            document.getElementById("preview").src = URL.createObjectURL(file);
            const form = new FormData();
            form.append("file", file);
            const response = await fetch("/predict", { method: "POST", body: form });
            const payload = await response.json();
            document.getElementById("result").textContent = JSON.stringify(payload, null, 2);
          }
        </script>
      </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
