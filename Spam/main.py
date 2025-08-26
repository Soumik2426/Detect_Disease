from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os  # ✅ Import to read environment variables

from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Dynamically Load Model Based on Environment Variable
MODEL_VERSION = os.getenv("MODEL_VERSION", "universal2")  # Default: universal2 (Production)
MODEL_PATH = f"/models/{MODEL_VERSION}.keras"  # Path inside container

try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)  # Load model at runtime
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}. Error: {e}")

CLASS_NAMES = ["Bacterial Blight", "Cercospora", "Healthy Coffee Leaf",
               "Healthy Sugarcane Leaf", "Mosaic", "RedRot",
               "Rust Coffee Leaf", "Rust Sugarcane Leaf", "Yellow"]

@app.get("/ping")
async def ping():
    return {"message": "pong"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {"Class": predicted_class, "Confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6700)