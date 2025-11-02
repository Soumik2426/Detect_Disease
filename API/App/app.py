import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI, File, UploadFile, Header, HTTPException
import numpy as np
import gdown
import requests
import json
from io import BytesIO
from PIL import Image
from starlette.middleware.cors import CORSMiddleware
import threading

try:
    import tensorflow as tf  # primary path: always use tensorflow.keras
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

# ------------------ Config ------------------
TF_SERVING_URL = os.getenv("TF_SERVING_URL", "http://localhost:8501/v1/models")
PRODUCTION_MODEL_NAME = os.getenv("PRODUCTION_MODEL_NAME", "Production_Model")
BETA_MODEL_NAME = os.getenv("BETA_MODEL_NAME", "Beta_Model")
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").lower()  # default local

# ------------------ Paths ------------------
# Base directory = Disease/API/App
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models directory = Disease/API/Models
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Models"))
os.makedirs(MODELS_DIR, exist_ok=True)

# Model file paths
PRODUCTION_MODEL_PATH = os.path.join(MODELS_DIR, "universal2.keras")
BETA_MODEL_PATH = os.path.join(MODELS_DIR, "universal3.keras")

# ------------------ Google Drive URLs ------------------
PRODUCTION_MODEL_URL = "https://drive.google.com/uc?export=download&id=1uROM2NGMpxnoTksBKkijem8IucOhC8dS"
BETA_MODEL_URL = "https://drive.google.com/uc?export=download&id=1p8BjCHcG_38eyH9C4locAT_OY9tantlB"

# ------------------ Auto-download if Missing ------------------
def download_if_missing(url, path):
    """Download file if not present."""
    if not os.path.exists(path):
        print(f"📦 Downloading model from {url} ...")
        try:
            gdown.download(url, path, quiet=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model download failed: {str(e)}")

        if not os.path.exists(path) or os.path.getsize(path) < 10000:
            raise HTTPException(status_code=500, detail=f"Model download failed or incomplete: {path}")
        print(f"✅ Model downloaded and saved to {path}")
    else:
        print(f"✅ Model already exists at {path}")

download_if_missing(PRODUCTION_MODEL_URL, PRODUCTION_MODEL_PATH)
download_if_missing(BETA_MODEL_URL, BETA_MODEL_PATH)

# ------------------ Model Loading ------------------
_local_models = {"production": None, "beta": None}
_model_lock = threading.Lock()

def _load_model_robust(model_path: str):
    """Load a Keras model trying multiple backends for compatibility."""
    last_error = None
    # 1. Try tf_keras.saving.load_model (TensorFlow 2.x)
    try:
        import tf_keras  # type: ignore
        if hasattr(tf_keras, "saving") and hasattr(tf_keras.saving, "load_model"):
            print(f"Loading model via tf_keras from {model_path}")
            return tf_keras.saving.load_model(model_path, compile=False)
    except Exception as e:
        last_error = e

    # 2. Try tensorflow.keras
    try:
        if tf is not None and hasattr(tf, "keras"):
            print(f"Loading model via tensorflow.keras from {model_path}")
            return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        last_error = e

    raise HTTPException(status_code=500, detail=f"Model load failed: {str(last_error)}")


# ------------------ FastAPI Routes ------------------
@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    """Readiness check & model paths."""
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

# ------------------ Image Preprocessing ------------------
INPUT_NORMALIZATION = os.getenv("INPUT_NORMALIZATION", "raw").lower()

def read_file_as_image(data) -> np.ndarray:
    """Convert uploaded image bytes into a numpy array."""
    img = Image.open(BytesIO(data)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img)
    if INPUT_NORMALIZATION == "raw":
        return arr.astype(np.float32)
    return (arr / 255.0).astype(np.float32)

# ------------------ Prediction Endpoint ------------------
@app.post("/models:predict")
async def predict(
    file: UploadFile = File(...),
    x_model_version: str = Header(None)
):
    """Image prediction endpoint."""
    is_beta = bool(x_model_version and x_model_version.lower() == "beta")
    img = read_file_as_image(await file.read())

    # Decide mode
    mode = INFERENCE_MODE
    if mode not in ("tfserving", "local"):
        mode = "tfserving"

    # Try TF Serving if enabled
    if mode == "tfserving":
        model_name = BETA_MODEL_NAME if is_beta else PRODUCTION_MODEL_NAME
        model_url = f"{TF_SERVING_URL}/{model_name}:predict"
        payload = json.dumps({"instances": [img.tolist()]})
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
                raise HTTPException(status_code=502, detail="TF Serving unreachable (INFERENCE_MODE=tfserving)")
            # fallback to local

    # Local Inference
    try:
        with _model_lock:
            key = "beta" if is_beta else "production"
            if _local_models[key] is None:
                model_path = BETA_MODEL_PATH if is_beta else PRODUCTION_MODEL_PATH
                if not os.path.isfile(model_path):
                    raise HTTPException(status_code=500, detail=f"Model file missing: {model_path}")
                _local_models[key] = _load_model_robust(model_path)

            model = _local_models[key]

        img_batch = np.expand_dims(img, 0)
        predictions = model.predict(img_batch)[0]

        predicted_class = CLASS_NAMES[int(np.argmax(predictions))]
        confidence = float(np.max(predictions))
        model_used = "beta" if is_beta else "production"

        return {"Class": predicted_class, "Confidence": confidence, "model": model_used}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local inference error: {str(e)}")

# ------------------ Main ------------------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    try:
        port = int(os.getenv("PORT", "8700"))
    except ValueError:
        port = 8700
    uvicorn.run("app:app", host=host, port=port)
