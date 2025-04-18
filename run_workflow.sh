#!/bin/bash

# Path to project directory
PROJECT_DIR=$(pwd)

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p plots

echo "====== Textile-Based Wearable Antenna Performance Prediction Workflow ======"
echo "Starting workflow execution at $(date)"

# Step 1: Generate synthetic data
echo ""
echo "Step 1: Generating synthetic data..."
python3 src/data/generate_synthetic_data.py --num_samples 200 --output_dir data/raw

# Step 2: Train the model
echo ""
echo "Step 2: Training the CNN model..."
python3 src/models/train_model.py --config config/model_config.json

# Step 3: Make predictions
echo ""
echo "Step 3: Making predictions on test data..."
# Find the most recent model directory
LATEST_MODEL=$(ls -td models/model_* | head -1)
echo "Using model: $LATEST_MODEL"
python3 src/models/predict_model.py --model_dir $LATEST_MODEL --params_file data/raw/antenna_params.csv --output_dir $LATEST_MODEL/predictions

echo ""
echo "Workflow completed at $(date)"
echo "Results are available in: $LATEST_MODEL/predictions"
echo "======================================================================" 