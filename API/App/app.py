from fastapi import FastAPI, File, UploadFile, Header, HTTPException
import numpy as np
import requests
import json
from io import BytesIO
from PIL import Image
from starlette.middleware.cors import CORSMiddleware

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

# TensorFlow Serving URLs
TF_SERVING_URL = "http://localhost:8501/v1/models"
PRODUCTION_MODEL_NAME = "Production_Model"
BETA_MODEL_NAME = "Beta_Model"


def read_file_as_image(data) -> np.ndarray:
    """Convert image bytes to a normalized NumPy array."""
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize pixel values
    return image.astype(np.float32)  # Ensure float32 type


@app.post("/models:predict")
async def predict(
    file: UploadFile = File(...),
    x_model_version: str = Header(None)  # Change header name for consistency
):
    """Handles image prediction requests."""

    # Determine which model to use
    model_name = BETA_MODEL_NAME if (x_model_version and x_model_version.lower() == "beta") else PRODUCTION_MODEL_NAME
    model_url = f"{TF_SERVING_URL}/{model_name}:predict"

    # Process image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0).tolist()

    # Prepare request to TensorFlow Serving
    payload = json.dumps({"instances": img_batch})
    headers = {"content-type": "application/json"}

    try:
        response = requests.post(model_url, data=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"TensorFlow Serving error: {str(e)}")

    # Extract prediction results
    result = response.json()
    predictions = result["predictions"][0]

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    return {
        "Class": predicted_class,
        "Confidence": float(confidence)
    }
