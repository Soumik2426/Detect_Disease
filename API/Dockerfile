# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for FastAPI
EXPOSE 7860

# Run the app
CMD ["uvicorn", "App.app:app", "--host", "0.0.0.0", "--port", "7860"]