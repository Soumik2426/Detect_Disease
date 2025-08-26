#!/bin/bash

# Define the paths
BETA_MODEL_PATH="/app/Models/universal3.keras"
NEW_BETA_MODEL_PATH="/app/Models/universal4.keras"
PRODUCTION_MODEL_SYMLINK="/app/production_model.keras"

# Update the symlink to point to the new beta model as the production model
echo "Promoting $NEW_BETA_MODEL_PATH to production..."
ln -sf $NEW_BETA_MODEL_PATH $PRODUCTION_MODEL_SYMLINK

# Reload the model in the FastAPI app (so the new model is loaded)
curl -X GET http://localhost:8700/reload-model

echo "Beta model $NEW_BETA_MODEL_PATH has been promoted to production!"
