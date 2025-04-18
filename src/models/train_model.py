import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import create_cnn_model, create_multioutput_model, get_callbacks
from data.data_processor import AntennaDataProcessor
from visualization.visualizer import AntennaVisualizer

def train_model(config_path):
    """
    Train a CNN model for antenna performance prediction.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration JSON file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], f"model_{timestamp}")
    model_dir = os.path.join(output_dir, "model")
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save config for reference
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Initialize data processor
    data_processor = AntennaDataProcessor(
        data_path=config["data_path"],
        target_size=(config["image_height"], config["image_width"]),
        test_size=config["test_size"],
        validation_size=config["validation_size"],
        random_state=config["random_state"]
    )
    
    # Prepare dataset
    dataset = data_processor.prepare_dataset(
        antenna_params_file=config["params_file"],
        preprocess=config["preprocess_data"]
    )
    
    # Save preprocessors
    if config["preprocess_data"]:
        data_processor.save_preprocessors(os.path.join(output_dir, "preprocessors"))
    
    # Get model input shape
    input_shape = dataset['X_train'][0].shape
    
    # Create model
    if config["model_type"] == "single_output":
        output_shape = dataset['y_train'][config["target_param"]].shape[1] \
            if len(dataset['y_train'][config["target_param"]].shape) > 1 \
            else 1
        
        model = create_cnn_model(
            input_shape=input_shape,
            output_shape=output_shape,
            dropout_rate=config["dropout_rate"],
            learning_rate=config["learning_rate"]
        )
        
        # Training data
        X_train = dataset['X_train']
        y_train = dataset['y_train'][config["target_param"]]
        X_val = dataset['X_val']
        y_val = dataset['y_val'][config["target_param"]]
        
    elif config["model_type"] == "multi_output":
        # For multi-output model, create a dictionary of output shapes
        output_shapes = {}
        for param in config["target_params"]:
            output_shapes[param] = dataset['y_train'][param].shape[1] \
                if len(dataset['y_train'][param].shape) > 1 \
                else 1
        
        model = create_multioutput_model(
            input_shape=input_shape,
            output_shapes=output_shapes,
            dropout_rate=config["dropout_rate"],
            learning_rate=config["learning_rate"]
        )
        
        # Training data
        X_train = dataset['X_train']
        y_train = {param: dataset['y_train'][param] for param in config["target_params"]}
        X_val = dataset['X_val']
        y_val = {param: dataset['y_val'][param] for param in config["target_params"]}
    
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    # Model checkpoint path
    checkpoint_path = os.path.join(model_dir, "best_model.h5")
    
    # Get callbacks
    callbacks = get_callbacks(
        checkpoint_path=checkpoint_path,
        patience=config["patience"]
    )
    
    # Add TensorBoard callback if specified
    if config["use_tensorboard"]:
        log_dir = os.path.join(output_dir, "logs", timestamp)
        os.makedirs(log_dir, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True
            )
        )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save(os.path.join(model_dir, "final_model.h5"))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    X_test = dataset['X_test']
    
    if config["model_type"] == "single_output":
        y_test = dataset['y_test'][config["target_param"]]
        y_pred = model.predict(X_test)
        
        # If using preprocessors, inverse transform
        if config["preprocess_data"]:
            predictions = {config["target_param"]: y_pred}
            inverse_predictions = data_processor.inverse_transform_predictions(predictions)
            y_pred = inverse_predictions[config["target_param"]]
            
            # Also inverse transform y_test
            test_data = {config["target_param"]: y_test}
            inverse_test = data_processor.inverse_transform_predictions(test_data)
            y_test = inverse_test[config["target_param"]]
        
        # Visualize results
        visualizer = AntennaVisualizer(save_dir=plot_dir)
        
        # Plot training history
        visualizer.plot_training_history(
            history,
            title=f"Training History - {config['target_param']}",
            save_as=f"training_history_{config['target_param']}.png"
        )
        
        # Plot prediction comparison
        metrics = visualizer.plot_prediction_comparison(
            y_test, y_pred,
            param_name=config["target_param"],
            save_as=f"prediction_comparison_{config['target_param']}.png"
        )
        
        # Save metrics
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
        
    else:  # multi-output
        y_test = {param: dataset['y_test'][param] for param in config["target_params"]}
        y_pred = model.predict(X_test)
        
        # If using preprocessors, inverse transform
        if config["preprocess_data"]:
            inverse_predictions = data_processor.inverse_transform_predictions(y_pred)
            y_pred = inverse_predictions
            
            # Also inverse transform y_test
            inverse_test = data_processor.inverse_transform_predictions(y_test)
            y_test = inverse_test
        
        # Visualize results
        visualizer = AntennaVisualizer(save_dir=plot_dir)
        
        # Plot training history
        visualizer.plot_training_history(
            history,
            title="Training History - Multi-output",
            save_as="training_history_multi.png"
        )
        
        # Plot prediction comparison for each output
        all_metrics = {}
        for param in config["target_params"]:
            metrics = visualizer.plot_prediction_comparison(
                y_test[param], y_pred[param],
                param_name=param,
                save_as=f"prediction_comparison_{param}.png"
            )
            all_metrics[param] = metrics
        
        # Save metrics
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        # Special plot for SAR if available
        if 'sar' in config["target_params"]:
            # Plot SAR distribution
            visualizer.plot_sar_distribution(
                y_pred['sar'].flatten(),
                threshold=1.6,  # Regulatory threshold
                save_as="sar_distribution.png"
            )
    
    print(f"\nTraining completed and results saved to {output_dir}")
    return model, output_dir

def main():
    parser = argparse.ArgumentParser(description='Train CNN model for antenna performance prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    args = parser.parse_args()
    
    train_model(args.config)

if __name__ == '__main__':
    main() 