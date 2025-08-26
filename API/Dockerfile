# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy necessary files from App folder
COPY App/requirements.txt /app/requirements.txt
COPY App/main.py /app/main.py

# Copy models from the API/Models folder
COPY Models/universal2.keras /app/Models/universal2.keras
COPY Models/universal3.keras /app/Models/universal3.keras

# Create a symlink pointing to the default production model
RUN ln -s /app/Models/universal2.keras /app/production_model.keras

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose port 8700
EXPOSE 8700

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8700"]
