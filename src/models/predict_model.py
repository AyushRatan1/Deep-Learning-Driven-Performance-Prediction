import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_processor import AntennaDataProcessor
from visualization.visualizer import AntennaVisualizer

def load_model_and_preprocessors(model_dir):
    """
    Load model and preprocessors from directory.
    
    Parameters:
    -----------
    model_dir : str
        Directory containing model and preprocessors
        
    Returns:
    --------
    model : tf.keras.Model
        Loaded model
    data_processor : AntennaDataProcessor
        Initialized data processor with loaded preprocessors
    config : dict
        Model configuration
    """
    # Load model
    model_path = os.path.join(model_dir, "model", "best_model.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model", "final_model.h5")
    
    model = tf.keras.models.load_model(model_path)
    
    # Load config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize data processor
    data_processor = AntennaDataProcessor(
        data_path=config["data_path"],
        target_size=(config["image_height"], config["image_width"]),
        test_size=config["test_size"],
        validation_size=config["validation_size"],
        random_state=config["random_state"]
    )
    
    # Load preprocessors if available
    preprocessors_path = os.path.join(model_dir, "preprocessors")
    if os.path.exists(preprocessors_path):
        data_processor.load_preprocessors(preprocessors_path)
    
    return model, data_processor, config

def predict_single_antenna(model, data_processor, pattern_file, config, output_dir=None):
    """
    Predict performance of a single antenna.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    data_processor : AntennaDataProcessor
        Initialized data processor
    pattern_file : str
        Path to antenna pattern file
    config : dict
        Model configuration
    output_dir : str, optional
        Directory to save prediction results
        
    Returns:
    --------
    predictions : dict
        Dictionary of predicted values for each output
    """
    # Load pattern file
    pattern = data_processor.load_radiation_pattern(pattern_file)
    
    # Convert to image format
    pattern_img = data_processor.convert_to_image(pattern)
    
    # Add batch dimension
    X = np.expand_dims(pattern_img, axis=0)
    
    # Preprocess if needed
    if data_processor.scaler_X is not None:
        X_flat = X.reshape(X.shape[0], -1)
        X_flat = data_processor.scaler_X.transform(X_flat)
        X = X_flat.reshape(X.shape)
    
    # Make prediction
    raw_predictions = model.predict(X)
    
    # Post-process predictions
    if config["model_type"] == "single_output":
        # Convert to dictionary format
        predictions = {config["target_param"]: raw_predictions}
    else:
        predictions = raw_predictions
    
    # Inverse transform if preprocessors exist
    if data_processor.scaler_y:
        predictions = data_processor.inverse_transform_predictions(predictions)
    
    # Save predictions if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        pred_dict = {}
        for key, pred in predictions.items():
            pred_dict[key] = pred.tolist()
            
        with open(os.path.join(output_dir, "predictions.json"), 'w') as f:
            json.dump(pred_dict, f, indent=4)
        
        # Visualize if visualizer available
        try:
            visualizer = AntennaVisualizer(save_dir=output_dir)
            
            # If S11 prediction is available, plot it
            if 's11' in predictions and isinstance(predictions['s11'], np.ndarray) and predictions['s11'].size > 1:
                # Assuming x-axis is frequency range from 2.2 to 2.6 GHz for ISM band 2.4 GHz
                # or 5.5 to 6.1 GHz for ISM band 5.8 GHz
                num_points = predictions['s11'].size
                
                # Choose frequency range based on number of points (heuristic)
                if num_points <= 100:
                    # Likely the 2.4 GHz band
                    freq_range = np.linspace(2.2, 2.6, num_points)
                    visualizer.plot_s_parameters(
                        freq_range, predictions['s11'][0],
                        title="Predicted S11 Parameter (2.4 GHz ISM Band)",
                        save_as="predicted_s11_2.4ghz.png"
                    )
                else:
                    # Likely the 5.8 GHz band
                    freq_range = np.linspace(5.5, 6.1, num_points)
                    visualizer.plot_s_parameters(
                        freq_range, predictions['s11'][0],
                        title="Predicted S11 Parameter (5.8 GHz ISM Band)",
                        save_as="predicted_s11_5.8ghz.png"
                    )
            
            # If SAR prediction is available, check if it's within threshold
            if 'sar' in predictions:
                sar_value = predictions['sar'][0][0]
                compliant = sar_value <= 1.6  # Regulatory threshold
                
                with open(os.path.join(output_dir, "sar_compliance.txt"), 'w') as f:
                    f.write(f"SAR value: {sar_value:.4f} W/kg\n")
                    f.write(f"Regulatory threshold: 1.6 W/kg\n")
                    f.write(f"Compliant: {compliant}\n")
        
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    return predictions

def predict_batch_antennas(model, data_processor, params_file, config, output_dir=None):
    """
    Predict performance for a batch of antennas.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    data_processor : AntennaDataProcessor
        Initialized data processor
    params_file : str
        Path to CSV file with antenna parameters
    config : dict
        Model configuration
    output_dir : str, optional
        Directory to save prediction results
        
    Returns:
    --------
    all_predictions : pd.DataFrame
        DataFrame with original parameters and predictions
    """
    # Load antenna parameters
    params_df = pd.read_csv(params_file)
    
    # Initialize lists for batch processing
    patterns = []
    ids = []
    
    # Process each antenna
    for idx, row in params_df.iterrows():
        pattern_file = os.path.join(config["data_path"], row['pattern_file'])
        pattern = data_processor.load_radiation_pattern(pattern_file)
        pattern_img = data_processor.convert_to_image(pattern)
        patterns.append(pattern_img)
        
        # Store ID or index for reference
        if 'id' in row:
            ids.append(row['id'])
        else:
            ids.append(idx)
    
    # Convert to numpy array
    X = np.array(patterns)
    
    # Preprocess if needed
    if data_processor.scaler_X is not None:
        X_flat = X.reshape(X.shape[0], -1)
        X_flat = data_processor.scaler_X.transform(X_flat)
        X = X_flat.reshape(X.shape)
    
    # Make predictions
    raw_predictions = model.predict(X)
    
    # Post-process predictions
    if config["model_type"] == "single_output":
        # Convert to dictionary format
        predictions = {config["target_param"]: raw_predictions}
    else:
        predictions = raw_predictions
    
    # Inverse transform if preprocessors exist
    if data_processor.scaler_y:
        predictions = data_processor.inverse_transform_predictions(predictions)
    
    # Create DataFrame with predictions
    pred_df = params_df.copy()
    
    # Add prediction columns
    for key, preds in predictions.items():
        if len(preds.shape) == 1 or preds.shape[1] == 1:
            # Single value prediction
            pred_df[f'predicted_{key}'] = preds.flatten()
        else:
            # Multiple values (like S11 curve) - store as JSON string
            for i, pred in enumerate(preds):
                pred_df.at[i, f'predicted_{key}'] = json.dumps(pred.tolist())
    
    # Save predictions if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save DataFrame
        pred_df.to_csv(os.path.join(output_dir, "batch_predictions.csv"), index=False)
        
        # Visualize if visualizer available
        try:
            visualizer = AntennaVisualizer(save_dir=output_dir)
            
            # Create plots based on available predictions
            
            # If gain prediction is available, plot distribution
            if 'predicted_gain' in pred_df.columns:
                gains = pred_df['predicted_gain'].values
                plt.figure(figsize=(10, 6))
                plt.hist(gains, bins=20)
                plt.axvline(x=3, color='r', linestyle='--', label='Min Acceptable (3 dBi)')
                plt.axvline(x=5, color='g', linestyle='--', label='Target (5 dBi)')
                plt.xlabel('Gain (dBi)')
                plt.ylabel('Count')
                plt.title('Distribution of Predicted Antenna Gains')
                plt.legend()
                plt.savefig(os.path.join(output_dir, "gain_distribution.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
            # If SAR prediction is available, plot distribution
            if 'predicted_sar' in pred_df.columns:
                sars = pred_df['predicted_sar'].values
                visualizer.plot_sar_distribution(
                    sars,
                    threshold=1.6,  # Regulatory threshold
                    title="Distribution of Predicted SAR Values",
                    save_as="sar_distribution.png"
                )
                
                # Calculate compliance percentage
                compliant = np.sum(sars <= 1.6)
                compliance_pct = (compliant / len(sars)) * 100
                
                with open(os.path.join(output_dir, "sar_compliance_summary.txt"), 'w') as f:
                    f.write(f"Total antennas: {len(sars)}\n")
                    f.write(f"Compliant antennas: {compliant} ({compliance_pct:.1f}%)\n")
                    f.write(f"Non-compliant antennas: {len(sars) - compliant} ({100 - compliance_pct:.1f}%)\n")
        
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    return pred_df

def main():
    parser = argparse.ArgumentParser(description='Predict antenna performance using trained CNN model')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the trained model')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save prediction results')
    
    # Either single file or batch file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pattern_file', type=str, help='Path to single antenna pattern file')
    group.add_argument('--params_file', type=str, help='Path to CSV file with multiple antenna parameters')
    
    args = parser.parse_args()
    
    # Load model and preprocessors
    model, data_processor, config = load_model_and_preprocessors(args.model_dir)
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Single prediction or batch prediction
    if args.pattern_file:
        predictions = predict_single_antenna(
            model, data_processor, args.pattern_file, config, args.output_dir
        )
        
        # Print predictions
        for key, pred in predictions.items():
            if pred.size == 1:
                print(f"{key}: {pred[0][0]:.4f}")
            else:
                print(f"{key}: Array of shape {pred.shape}")
    
    else:  # Batch prediction
        pred_df = predict_batch_antennas(
            model, data_processor, args.params_file, config, args.output_dir
        )
        
        # Print summary
        for col in pred_df.columns:
            if col.startswith('predicted_'):
                # Check if this column contains scalar values
                try:
                    values = pred_df[col].astype(float)
                    print(f"{col}:")
                    print(f"  Min: {values.min():.4f}")
                    print(f"  Max: {values.max():.4f}")
                    print(f"  Mean: {values.mean():.4f}")
                    print(f"  Std: {values.std():.4f}")
                except:
                    # Column contains non-scalar values (like arrays)
                    print(f"{col}: Non-scalar values")
    
    print("Prediction completed.")

if __name__ == '__main__':
    main() 