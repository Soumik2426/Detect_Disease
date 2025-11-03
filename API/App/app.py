import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import gdown
import json
from io import BytesIO
from PIL import Image
from starlette.middleware.cors import CORSMiddleware
import threading
import gc  # 🧹 for memory cleanup

try:
    import tensorflow as tf
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
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").lower()

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Models"))
os.makedirs(MODELS_DIR, exist_ok=True)

PRODUCTION_MODEL_PATH = os.path.join(MODELS_DIR, "universal2.keras")
PRODUCTION_MODEL_URL = "https://drive.google.com/uc?export=download&id=1uROM2NGMpxnoTksBKkijem8IucOhC8dS"

# ------------------ Download if Missing ------------------
def download_if_missing(url, path):
    if not os.path.exists(path):
        print(f"📦 Downloading model from {url} ...")
        try:
            gdown.download(url, path, quiet=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model download failed: {str(e)}")

        if not os.path.exists(path) or os.path.getsize(path) < 10_000:
            raise HTTPException(status_code=500, detail=f"Model download failed or incomplete: {path}")
        print(f"✅ Model downloaded and saved to {path}")
    else:
        print(f"✅ Model already exists at {path}")

download_if_missing(PRODUCTION_MODEL_URL, PRODUCTION_MODEL_PATH)

# ------------------ Model Loading ------------------
_model_lock = threading.Lock()

def _load_model_robust(model_path: str):
    last_error = None
    try:
        import tf_keras
        if hasattr(tf_keras, "saving") and hasattr(tf_keras.saving, "load_model"):
            print(f"Loading model via tf_keras from {model_path}")
            return tf_keras.saving.load_model(model_path, compile=False)
    except Exception as e:
        last_error = e

    try:
        if tf is not None and hasattr(tf, "keras"):
            print(f"Loading model via tensorflow.keras from {model_path}")
            return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        last_error = e

    raise HTTPException(status_code=500, detail=f"Local model load failed: {str(last_error)}")

# ------------------ Routes ------------------
@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/ready")
async def ready():
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
    img = Image.open(BytesIO(data)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img)
    if INPUT_NORMALIZATION == "raw":
        return arr.astype(np.float32)
    return (arr / 255.0).astype(np.float32)

# ------------------ Lazy-Loaded Prediction ------------------
@app.post("/models:predict")
async def predict(file: UploadFile = File(...)):
    """
    Load model only when needed, predict, then free memory (Render-friendly).
    """
    img = read_file_as_image(await file.read())

    with _model_lock:
        if not os.path.isfile(PRODUCTION_MODEL_PATH):
            raise HTTPException(status_code=500, detail=f"Model file missing: {PRODUCTION_MODEL_PATH}")

        # 🧠 Load model temporarily
        model = _load_model_robust(PRODUCTION_MODEL_PATH)

        try:
            img_batch = np.expand_dims(img, 0)
            predictions = model.predict(img_batch)[0]
            predicted_class = CLASS_NAMES[int(np.argmax(predictions))]
            confidence = float(np.max(predictions))
        finally:
            # 🧹 Free up memory immediately
            del model
            gc.collect()

    return {"Class": predicted_class, "Confidence": confidence}