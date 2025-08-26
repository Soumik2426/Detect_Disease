#!/bin/bash

# Define the paths
NEW_MODEL_PATH="/app/Models/universal5.keras"
BETA_MODEL_SYMLINK="/app/Models/universal3.keras"

# Create a symlink to the new model as the beta model
echo "Adding $NEW_MODEL_PATH as the new beta model..."
ln -sf $NEW_MODEL_PATH $BETA_MODEL_SYMLINK

# Reload the model in the FastAPI app
curl -X GET http://localhost:8700/reload-model

echo "$NEW_MODEL_PATH has been added as the new beta model!"
