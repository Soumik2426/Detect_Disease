from fastapi import FastAPI, File, UploadFile, Header
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import threading
import json
import uuid
from starlette.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for demo purposes)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Path to models inside the 'Disease/Models' folder
MODEL_SYMLINK_PATH = "/app/Models/universal2.keras"  # Production model path
BETA_MODEL_PATH = "/app/Models/universal3.keras"  # Beta model path

# Load the production model by default
MODEL = tf.keras.models.load_model(MODEL_SYMLINK_PATH)

# Define class names (replace with your actual class names)
CLASS_NAMES = [
    "Bacterial Blight", "Cercospora", "Healthy Coffee Leaf", "Healthy Sugarcane Leaf",
    "Mosaic", "RedRot", "Rust Coffee Leaf", "Rust Sugarcane Leaf", "Yellow"
]

# Lock for thread-safety during model reloading
model_lock = threading.Lock()

# Path to store API keys
API_KEYS_FILE = "/app/api_keys.json"

# Function to generate and store the API key
def generate_api_key():
    # Generate a new API key (UUID)
    api_key = str(uuid.uuid4())

    # Load existing keys if the file exists
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as file:
            api_keys = json.load(file)
    else:
        api_keys = {}

    # Save the new API key into the dictionary
    api_keys[api_key] = "your_access_level_or_description"  # Add any other information here if needed

    # Save the updated API keys back into the file
    with open(API_KEYS_FILE, "w") as file:
        json.dump(api_keys, file)

    return api_key

@app.get("/ping")
async def ping():
    return "pong"

def read_file_as_image(data) -> np.ndarray:
    """Converts the uploaded file to an image and resizes it."""
    image = Image.open(BytesIO(data))
    image = image.resize((224, 224))  # Resize image to 224x224 for model
    image = np.array(image)
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    use_beta_model: str = Header(None),  # Use header to access the beta model
    api_key: str = Header(None)  # API Key for authorization
):
    # Verify the API key
    if not verify_api_key(api_key):
        return {"error": "Invalid API key"}

    global MODEL

    # Select model based on header (default is production model)
    if use_beta_model == "true":
        with model_lock:
            MODEL = tf.keras.models.load_model(BETA_MODEL_PATH)  # Switch to beta model
    else:
        with model_lock:
            MODEL = tf.keras.models.load_model(MODEL_SYMLINK_PATH)  # Switch to production model

    # Process the image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # Add batch dimension

    # Make predictions using the selected model
    predictions = MODEL.predict(img_batch)

    # Get the predicted class and confidence
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        "Class": predicted_class,
        "Confidence": float(confidence)
    }

@app.get("/reload-model")
async def reload_model():
    """Reloads the model dynamically from the symlink without restarting the API."""
    global MODEL
    with model_lock:
        MODEL = tf.keras.models.load_model(MODEL_SYMLINK_PATH)
    return {"message": "Model reloaded successfully"}

@app.post("/generate-api-key")
async def generate_api_key_endpoint():
    # Generate and store the API key
    api_key = generate_api_key()
    return {"api_key": api_key}

def verify_api_key(api_key: str) -> bool:
    """Verify if the provided API key is valid."""
    if not api_key:
        return False

    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as file:
            api_keys = json.load(file)
            return api_key in api_keys
    return False

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8700)
