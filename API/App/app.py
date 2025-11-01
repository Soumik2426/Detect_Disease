import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI, File, UploadFile, Header, HTTPException
import numpy as np
import os
import gdown
import requests
import json
from io import BytesIO
from PIL import Image
from starlette.middleware.cors import CORSMiddleware
import os
import threading
try:
    import tensorflow as tf  # primary path: always use tensorflow.keras
except Exception:
    tf = None

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class names for prediction output
CLASS_NAMES = [
    "Bacterial Blight", "Cercospora", "Healthy Coffee Leaf", "Healthy Sugarcane Leaf",
    "Mosaic", "RedRot", "Rust Coffee Leaf", "Rust Sugarcane Leaf", "Yellow"
]

# TensorFlow Serving configuration (override via environment variables)
TF_SERVING_URL = os.getenv("TF_SERVING_URL", "http://localhost:8501/v1/models")
PRODUCTION_MODEL_NAME = os.getenv("PRODUCTION_MODEL_NAME", "Production_Model")
BETA_MODEL_NAME = os.getenv("BETA_MODEL_NAME", "Beta_Model")
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").lower()  # default: local for out-of-the-box usage

# Local model paths (used when INFERENCE_MODE == 'local' or TF Serving unavailable)
_default_models_dir = "/app/Models" if os.path.isdir("/app/Models") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Models"))
PRODUCTION_MODEL_PATH = os.getenv("PRODUCTION_MODEL_PATH", os.path.join(_default_models_dir, "universal2.keras"))
BETA_MODEL_PATH = os.getenv("BETA_MODEL_PATH", os.path.join(_default_models_dir, "universal3.keras"))

# --- Auto-download models from Google Drive if missing ---
import requests

def download_if_missing(url, path):
    if not os.path.exists(path):
        print(f"📦 Downloading model from {url} ...")
        gdown.download(url, path, quiet=False, fuzzy=True)
        if not os.path.exists(path) or os.path.getsize(path) < 10000:
            raise Exception("❌ Model download failed or file too small. Check Google Drive permissions.")
        print(f"✅ Model downloaded and saved to {path}")


# Replace YOUR_FILE_ID_1 and YOUR_FILE_ID_2 with actual Drive IDs
PRODUCTION_MODEL_URL = "https://drive.google.com/file/d/1uROM2NGMpxnoTksBKkijem8IucOhC8dS/view?usp=sharing"
BETA_MODEL_URL = "https://drive.google.com/file/d/1p8BjCHcG_38eyH9C4locAT_OY9tantlB/view?usp=sharing"

download_if_missing(PRODUCTION_MODEL_URL, PRODUCTION_MODEL_PATH)
download_if_missing(BETA_MODEL_URL, BETA_MODEL_PATH)

# Lazy-loaded local models with a lock
_local_models = {"production": None, "beta": None}
_model_lock = threading.Lock()


def _load_model_robust(model_path: str):
    """Load a Keras model trying multiple backends for maximum compatibility."""
    last_error = None
    # 1) Prefer tf_keras.saving.load_model (matches TF-Keras 2.x format)
    try:
        import tf_keras  # type: ignore
        if hasattr(tf_keras, "saving") and hasattr(tf_keras.saving, "load_model"):
            return tf_keras.saving.load_model(model_path, compile=False)
    except Exception as e:
        last_error = e
    # 2) tensorflow.keras
    try:
        if tf is not None and hasattr(tf, "keras"):
            return tf.keras.models.load_model(model_path, compile=False)  # type: ignore[attr-defined]
    except Exception as e:
        last_error = e
    raise HTTPException(status_code=500, detail=f"Local model load failed: {str(last_error)}")


if __name__ == "__main__":
    # Allow running via: python app.py
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    try:
        port = int(os.getenv("PORT", "8700"))
    except ValueError:
        port = 8700
    uvicorn.run("app:app", host=host, port=port)

@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness and routing info."""
    return {
        "status": "ready",
        "inference_mode": INFERENCE_MODE,
        "tf_serving_url": TF_SERVING_URL,
        "production_model": PRODUCTION_MODEL_NAME,
        "beta_model": BETA_MODEL_NAME,
        "local_production_model_exists": os.path.isfile(PRODUCTION_MODEL_PATH),
        "local_beta_model_exists": os.path.isfile(BETA_MODEL_PATH),
        "production_model_path": PRODUCTION_MODEL_PATH,
        "beta_model_path": BETA_MODEL_PATH
    }


INPUT_NORMALIZATION = os.getenv("INPUT_NORMALIZATION", "raw").lower()  # default: raw as per best results

def read_file_as_image(data) -> np.ndarray:
    """Convert image bytes to an array with configurable normalization."""
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))  # Resize to match model input
    arr = np.array(image)
    if INPUT_NORMALIZATION == "raw":
        return arr.astype(np.float32)
    # default to unit scaling [0,1]
    return (arr / 255.0).astype(np.float32)


@app.post("/models:predict")
async def predict(
    file: UploadFile = File(...),
    x_model_version: str = Header(None)  # Change header name for consistency
):
    """Handles image prediction requests."""

    # Determine requested target (beta vs production)
    is_beta = bool(x_model_version and x_model_version.lower() == "beta")

    # Process image once
    image = read_file_as_image(await file.read())

    # Route selection
    mode = INFERENCE_MODE
    if mode not in ("tfserving", "local"):
        mode = "tfserving"  # try tfserving first in auto mode

    # Try TF Serving first if requested
    if mode == "tfserving":
        model_name = BETA_MODEL_NAME if is_beta else PRODUCTION_MODEL_NAME
        model_url = f"{TF_SERVING_URL}/{model_name}:predict"
        payload = json.dumps({"instances": [image.tolist()]})
        headers = {"content-type": "application/json"}
        try:
            response = requests.post(model_url, data=payload, headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()
            predictions = result["predictions"][0]
            predicted_class = CLASS_NAMES[int(np.argmax(predictions))]
            confidence = float(np.max(predictions))
            model_used = "beta" if is_beta else "production"
            return {"Class": predicted_class, "Confidence": confidence, "model": model_used}
        except requests.exceptions.RequestException:
            if INFERENCE_MODE == "tfserving":
                raise HTTPException(status_code=502, detail="TensorFlow Serving unreachable and local fallback disabled (INFERENCE_MODE=tfserving)")
            # fall through to local

    # Local inference path
    try:
        with _model_lock:
            key = "beta" if is_beta else "production"
            if _local_models[key] is None:
                model_path = BETA_MODEL_PATH if is_beta else PRODUCTION_MODEL_PATH
                if not os.path.isfile(model_path):
                    raise HTTPException(status_code=500, detail=f"Local model file not found: {model_path}")
                _local_models[key] = _load_model_robust(model_path)

            model = _local_models[key]

        # Model expects batch dimension
        img_batch = np.expand_dims(image, 0)
        predictions = model.predict(img_batch)
        predictions = predictions[0]

        predicted_class = CLASS_NAMES[int(np.argmax(predictions))]
        confidence = float(np.max(predictions))
        model_used = "beta" if is_beta else "production"
        return {"Class": predicted_class, "Confidence": confidence, "model": model_used}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local inference error: {str(e)}")
