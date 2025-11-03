import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import gdown
import requests
import json
from io import BytesIO
from PIL import Image
from starlette.middleware.cors import CORSMiddleware
import threading

# tensorflow import optional (we handle missing gracefully)
try:
    import tensorflow as tf  # primary path: tensorflow.keras
except Exception:
    tf = None

app = FastAPI()

# ------------------ Enable CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Class Labels ------------------
CLASS_NAMES = [
    "Bacterial Blight", "Cercospora", "Healthy Coffee Leaf", "Healthy Sugarcane Leaf",
    "Mosaic", "RedRot", "Rust Coffee Leaf", "Rust Sugarcane Leaf", "Yellow"
]

# ------------------ Config (kept for compatibility) ------------------
# TF Serving left here for reference but we will NOT call it in Option 2
TF_SERVING_URL = os.getenv("TF_SERVING_URL", "http://localhost:8501/v1/models")
PRODUCTION_MODEL_NAME = os.getenv("PRODUCTION_MODEL_NAME", "Production_Model")
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").lower()  # default local

# ------------------ Paths ------------------
# Base directory = Disease/API/App
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models directory = Disease/API/Models
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Models"))
os.makedirs(MODELS_DIR, exist_ok=True)

# Production model path (keep the same model you were using)
PRODUCTION_MODEL_PATH = os.path.join(MODELS_DIR, "universal2.keras")

# ------------------ Google Drive URL for production model ------------------
PRODUCTION_MODEL_URL = "https://drive.google.com/uc?export=download&id=1uROM2NGMpxnoTksBKkijem8IucOhC8dS"

# ------------------ Auto-download if Missing (production only) ------------------
def download_if_missing(url, path):
    """Download file if not present (production model only)."""
    if not os.path.exists(path):
        print(f"📦 Downloading model from {url} ...")
        try:
            # gdown will follow Drive confirm tokens when fuzzy=True, but we keep default behavior.
            gdown.download(url, path, quiet=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model download failed: {str(e)}")

        # Basic sanity check on size (very small file likely means HTML error page)
        if not os.path.exists(path) or os.path.getsize(path) < 10_000:
            raise HTTPException(status_code=500, detail=f"Model download failed or incomplete: {path}")
        print(f"✅ Model downloaded and saved to {path}")
    else:
        print(f"✅ Model already exists at {path}")

# Ensure production model present
download_if_missing(PRODUCTION_MODEL_URL, PRODUCTION_MODEL_PATH)

# ------------------ Model Loading ------------------
_local_model = None
_model_lock = threading.Lock()

def _load_model_robust(model_path: str):
    """Load a Keras model trying multiple backends for compatibility (same approach you used)."""
    last_error = None
    # 1) try tf_keras.saving.load_model (if available)
    try:
        import tf_keras  # type: ignore
        if hasattr(tf_keras, "saving") and hasattr(tf_keras.saving, "load_model"):
            print(f"Loading model via tf_keras from {model_path}")
            return tf_keras.saving.load_model(model_path, compile=False)
    except Exception as e:
        last_error = e

    # 2) try tensorflow.keras
    try:
        if tf is not None and hasattr(tf, "keras"):
            print(f"Loading model via tensorflow.keras from {model_path}")
            return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        last_error = e

    # If both fail, propagate a helpful error
    raise HTTPException(status_code=500, detail=f"Local model load failed: {str(last_error)}")

# ------------------ FastAPI Routes ------------------
@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    """Readiness check & model path info."""
    return {
        "status": "ready",
        "inference_mode": INFERENCE_MODE,
        "tf_serving_url": TF_SERVING_URL,
        "production_model": PRODUCTION_MODEL_NAME,
        "local_production_model_exists": os.path.isfile(PRODUCTION_MODEL_PATH),
        "production_model_path": PRODUCTION_MODEL_PATH
    }

# ------------------ Image Preprocessing ------------------
INPUT_NORMALIZATION = os.getenv("INPUT_NORMALIZATION", "raw").lower()

def read_file_as_image(data) -> np.ndarray:
    """Convert uploaded image bytes into a numpy array using same resizing + normalization logic."""
    img = Image.open(BytesIO(data)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img)
    if INPUT_NORMALIZATION == "raw":
        # preserve original behaviour you had
        return arr.astype(np.float32)
    # else normalized to [0,1]
    return (arr / 255.0).astype(np.float32)

# ------------------ Prediction Endpoint (Option 2: local production only) ------------------
@app.post("/models:predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict using only the production model (lazy-loaded).
    Response intentionally does NOT include which model was used.
    """
    global _local_model

    # Read image bytes and preprocess (keeps your normalization logic)
    img = read_file_as_image(await file.read())

    # Lazy-load the single production model (thread-safe)
    if _local_model is None:
        with _model_lock:
            if _local_model is None:
                if not os.path.isfile(PRODUCTION_MODEL_PATH):
                    raise HTTPException(status_code=500, detail=f"Model file missing: {PRODUCTION_MODEL_PATH}")
                _local_model = _load_model_robust(PRODUCTION_MODEL_PATH)

    try:
        # Model expects batch dim
        img_batch = np.expand_dims(img, 0)
        predictions = _local_model.predict(img_batch)[0]

        predicted_class = CLASS_NAMES[int(np.argmax(predictions))]
        confidence = float(np.max(predictions))

        # Return only class + confidence (no model field)
        return {"Class": predicted_class, "Confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local inference error: {str(e)}")

# ------------------ Main ------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
