# Use official lightweight Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . /app

# Upgrade pip tools first
RUN pip install --upgrade pip setuptools wheel

# Install all locked dependencies
RUN pip install --no-cache-dir -r requirements.lock

# Expose the FastAPI port
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "App.app:app", "--host", "0.0.0.0", "--port", "7860"]