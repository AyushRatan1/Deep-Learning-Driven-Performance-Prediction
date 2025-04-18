#!/bin/bash

# Path to project directory
PROJECT_DIR=$(pwd)

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p plots

echo "====== Textile-Based Wearable Antenna Synthetic Data Generation ======"
echo "Starting data generation at $(date)"

# Install basic required packages
echo ""
echo "Installing required packages..."
python3 -m pip install numpy pandas h5py

# Step 1: Generate synthetic data
echo ""
echo "Generating synthetic data..."
python3 src/data/generate_synthetic_data.py --num_samples 50 --output_dir data/raw

echo ""
echo "Data generation completed at $(date)"
echo "Generated data is available in data/raw directory"
echo "========================================================================"

# Show summary of generated data
echo ""
echo "Summary of generated antenna parameters:"
head -n 5 data/raw/antenna_params.csv
echo "..."
echo "Total antennas generated: $(wc -l < data/raw/antenna_params.csv)" 