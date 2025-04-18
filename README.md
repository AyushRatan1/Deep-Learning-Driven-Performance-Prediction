# Deep Learning-Driven Performance Prediction of Textile-Based Wearable Antennas

This project implements a Convolutional Neural Network (CNN) approach to predict the performance of textile-based wearable antennas operating in ISM bands (2.4 GHz and 5.8 GHz) for healthcare applications.

## Project Overview

The increasing adoption of wearable technologies in healthcare has created a demand for efficient, safe, and flexible antenna designs. This project focuses on using deep learning to predict key antenna performance parameters, reducing the time and complexity associated with traditional full-wave electromagnetic simulations.

### Key Features

- **CNN-based performance prediction** for textile-based wearable antennas
- **Multi-output prediction** of S11 (return loss), gain, radiation patterns, and SAR
- **Synthetic data generation** for training and testing
- **Performance visualization** and analysis tools
- **SAR compliance checking** against regulatory thresholds

### Target Performance Metrics

The model aims to predict antenna designs meeting the following specifications:
- **Gain**: 3-5 dBi
- **SAR**: Below 1.6 W/kg (regulatory threshold)
- **Mechanical bending radius**: > 10 mm (for comfort and adaptability)

## Project Structure

```
.
├── config/               # Configuration files
│   └── model_config.json # Model training configuration
├── data/                 # Dataset storage
│   ├── raw/              # Raw simulation and measurement data
│   └── processed/        # Processed data for model training
├── models/               # Saved model checkpoints
├── notebooks/            # Jupyter notebooks for experimentation
│   └── antenna_prediction_example.ipynb # Example workflow notebook
├── plots/                # Visualization outputs
├── src/                  # Source code
│   ├── data/             # Data processing scripts
│   │   ├── data_processor.py       # Data preprocessing module
│   │   └── generate_synthetic_data.py # Synthetic data generator
│   ├── features/         # Feature engineering 
│   ├── models/           # Model architectures
│   │   ├── cnn_model.py  # CNN model definition
│   │   ├── train_model.py # Training script
│   │   └── predict_model.py # Prediction script
│   └── visualization/    # Visualization utilities
│       └── visualizer.py # Visualization module
├── tests/                # Test cases
│   └── test_model.py     # Model architecture tests
├── run_workflow.sh       # End-to-end workflow script
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Installation and Setup

1. Clone this repository:
```bash
git clone https://github.com/username/textile-antenna-prediction.git
cd textile-antenna-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories (if not already present):
```bash
mkdir -p data/raw data/processed models plots
```

## Usage

### Quick Start

Run the entire workflow (data generation, model training, and prediction) using:

```bash
./run_workflow.sh
```

### Step-by-Step Workflow

1. **Generate synthetic data**:
```bash
python src/data/generate_synthetic_data.py --num_samples 200 --output_dir data/raw
```

2. **Train the model**:
```bash
python src/models/train_model.py --config config/model_config.json
```

3. **Make predictions**:
```bash
python src/models/predict_model.py --model_dir models/model_YYYYMMDD_HHMMSS --params_file data/raw/antenna_params.csv --output_dir predictions
```

### Jupyter Notebook

Explore the workflow interactively using the provided Jupyter notebook:

```bash
jupyter notebook notebooks/antenna_prediction_example.ipynb
```

## Technical Approach

### Deep Learning Model

The project implements two types of CNN models:
1. **Single-output model**: Predicts a specific antenna parameter (e.g., gain or SAR)
2. **Multi-output model**: Simultaneously predicts multiple parameters (S11, gain, SAR)

The CNN architecture consists of:
- Convolutional layers for feature extraction
- Batch normalization for training stability
- Dropout layers for regularization
- Dense layers for parameter prediction

### Data Representation

Antenna radiation patterns are converted to image-like representations for CNN processing. S-parameters and other performance metrics are processed and scaled appropriately before training.

### Performance Metrics

The model's performance is evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² score

## Applications

This project is particularly useful for:

1. **Healthcare Wearables**: Designing antennas for continuous health monitoring devices
2. **Telemedicine**: Reliable communication in remote healthcare applications
3. **Positioning Systems**: Location tracking for patient monitoring
4. **Rapid Prototyping**: Accelerated design cycle for wearable antenna development

## Future Work

- Integration with electromagnetic simulation software for validation
- Extended model for different antenna types (beyond patch and monopole)
- Real-time adaptive tuning of antennas in dynamic body-worn environments
- Transfer learning from simulation data to improve prediction on real measurements
- Web interface for antenna designers to predict performance without simulation

## References

1. IEEE standards for SAR limits in body-worn devices
2. ISM band regulations (2.4 GHz and 5.8 GHz)
3. Literature on textile-based wearable antennas
4. Deep learning approaches for electromagnetic design optimization

## License

MIT

## Acknowledgements

This project is inspired by the growing need for efficient design methodologies in the field of wearable antennas for healthcare applications. 