# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy necessary files (fix paths for Docker context)
COPY API/App/app.py /app/app.py
COPY requirements.lock /app/requirements.lock

# Copy models from the API/Models folder (fix path)
COPY API/Models/universal2.keras /app/Models/universal2.keras
COPY API/Models/universal3.keras /app/Models/universal3.keras

# Create a symlink pointing to the default production model
RUN ln -s /app/Models/universal2.keras /app/production_model.keras

# Install dependencies from lockfile for reproducible builds
RUN pip install --no-cache-dir -r /app/requirements.lock

# Expose port 8700
EXPOSE 8700

# Default to local inference so the app works out-of-the-box
ENV INFERENCE_MODE="local"
ENV PRODUCTION_MODEL_PATH="/app/Models/universal2.keras"
ENV BETA_MODEL_PATH="/app/Models/universal3.keras"
ENV PRODUCTION_MODEL_NAME="Production_Model"
ENV BETA_MODEL_NAME="Beta_Model"

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8700"]
