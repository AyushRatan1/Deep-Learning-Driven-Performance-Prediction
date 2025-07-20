import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import create_cnn_model, create_multioutput_model, create_physics_informed_model, get_callbacks
from data.data_processor import AntennaDataProcessor
from visualization.visualizer import AntennaVisualizer

def train_model(config_path):
    """
    Train an enhanced CNN model for antenna performance prediction with improved accuracy.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration JSON file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("="*80)
    print("ENHANCED ANTENNA PERFORMANCE PREDICTION TRAINING")
    print("="*80)
    print(f"Configuration: {config_path}")
    print(f"Model type: {config['model_type']}")
    print(f"Target parameters: {config.get('target_params', [config.get('target_param', 'N/A')])}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], f"model_{timestamp}")
    model_dir = os.path.join(output_dir, "model")
    plot_dir = os.path.join(output_dir, "plots")
    log_dir = os.path.join(output_dir, "logs")
    
    for dir_path in [output_dir, model_dir, plot_dir, log_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Save config for reference
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Output directory: {output_dir}")
    
    # Initialize data processor with enhanced settings
    data_processor = AntennaDataProcessor(
        data_path=config["data_path"],
        target_size=(config["image_height"], config["image_width"]),
        test_size=config["test_size"],
        validation_size=config["validation_size"],
        random_state=config["random_state"]
    )
    
    # Prepare dataset with augmentation for better generalization
    print("\n" + "="*50)
    print("DATA PREPARATION")
    print("="*50)
    
    dataset = data_processor.prepare_dataset(
        antenna_params_file=config["params_file"],
        preprocess=config["preprocess_data"],
        augment=config.get("augment_data", True)
    )
    
    # Save preprocessors
    if config["preprocess_data"]:
        preprocessors_dir = os.path.join(output_dir, "preprocessors")
        data_processor.save_preprocessors(preprocessors_dir)
        print(f"Preprocessors saved to: {preprocessors_dir}")
    
    # Get model input shape
    input_shape = dataset['X_train'][0].shape
    print(f"Input shape: {input_shape}")
    
    # Create model with enhanced architecture
    print("\n" + "="*50)
    print("MODEL CREATION")
    print("="*50)
    
    if config["model_type"] == "single_output":
        output_shape = dataset['y_train'][config["target_param"]].shape[1] \
            if len(dataset['y_train'][config["target_param"]].shape) > 1 \
            else 1
        
        print(f"Creating single-output model for {config['target_param']}")
        print(f"Output shape: {output_shape}")
        
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
            shape = dataset['y_train'][param].shape[1] \
                if len(dataset['y_train'][param].shape) > 1 else 1
            output_shapes[param] = shape
            
        print(f"Creating multi-output model")
        print(f"Output shapes: {output_shapes}")
        
        # Choose enhanced architecture based on configuration
        model_architecture = config.get("model_architecture", "enhanced")
        
        if model_architecture == "physics_informed":
            print("Using physics-informed neural network")
            model = create_physics_informed_model(
                input_shape=input_shape,
                output_shapes=output_shapes,
                dropout_rate=config["dropout_rate"],
                learning_rate=config["learning_rate"]
            )
        else:
            print("Using enhanced multi-output CNN")
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
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Model checkpoint path
    checkpoint_path = os.path.join(model_dir, "best_model.h5")
    
    # Get enhanced callbacks
    monitor_metric = 'val_loss' if config["model_type"] == "single_output" else 'val_loss'
    callbacks = get_callbacks(
        checkpoint_path=checkpoint_path,
        patience=config.get("patience", 20),
        monitor=monitor_metric
    )
    
    # Add TensorBoard callback if specified
    if config.get("use_tensorboard", True):
        tb_log_dir = os.path.join(log_dir, timestamp)
        os.makedirs(tb_log_dir, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=tb_log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        )
        print(f"TensorBoard logs: {tb_log_dir}")
    
    # Train the model
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Dropout rate: {config['dropout_rate']}")
    
    # Custom training with better monitoring
    class TrainingMonitor(tf.keras.callbacks.Callback):
        def __init__(self):
            self.best_val_loss = float('inf')
            self.best_epoch = 0
        
        def on_epoch_end(self, epoch, logs=None):
            val_loss = logs.get('val_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch}: val_loss = {val_loss:.6f} (best: {self.best_val_loss:.6f} @ epoch {self.best_epoch})")
                
                # Print per-parameter metrics for multi-output models
                if config["model_type"] == "multi_output":
                    for param in config["target_params"]:
                        val_param_loss = logs.get(f'val_{param}_loss', 0)
                        val_param_mae = logs.get(f'val_{param}_mae', 0)
                        print(f"  {param}: loss={val_param_loss:.6f}, mae={val_param_mae:.6f}")
    
    monitor = TrainingMonitor()
    callbacks.append(monitor)
    
    try:
        history = model.fit(
            X_train, y_train,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {monitor.best_val_loss:.6f} at epoch {monitor.best_epoch}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    # Save the final model
    final_model_path = os.path.join(model_dir, "final_model.h5")
    model.save(final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Comprehensive evaluation on test set
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    X_test = dataset['X_test']
    
    # Load the best model for evaluation
    try:
        best_model = tf.keras.models.load_model(checkpoint_path)
        print(f"Loaded best model from: {checkpoint_path}")
    except:
        print("Could not load best model, using current model")
        best_model = model
    
    if config["model_type"] == "single_output":
        y_test = dataset['y_test'][config["target_param"]]
        y_pred = best_model.predict(X_test, verbose=0)
        
        # If using preprocessors, inverse transform
        if config["preprocess_data"]:
            predictions = {config["target_param"]: y_pred}
            inverse_predictions = data_processor.inverse_transform_predictions(predictions)
            y_pred = inverse_predictions[config["target_param"]]
            
            # Also inverse transform y_test
            test_data = {config["target_param"]: y_test}
            inverse_test = data_processor.inverse_transform_predictions(test_data)
            y_test = inverse_test[config["target_param"]]
        
        # Calculate comprehensive metrics
        y_test_flat = y_test.flatten() if y_test.ndim > 1 else y_test
        y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
        
        mse = mean_squared_error(y_test_flat, y_pred_flat)
        mae = mean_absolute_error(y_test_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_flat, y_pred_flat)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((y_test_flat - y_pred_flat) / np.maximum(np.abs(y_test_flat), 1e-8))) * 100
        
        print(f"\nSingle-Output Model Results ({config['target_param']}):")
        print("-" * 50)
        print(f"Mean Squared Error (MSE):     {mse:.6f}")
        print(f"Root Mean Squared Error:      {rmse:.6f}")
        print(f"Mean Absolute Error (MAE):    {mae:.6f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        print(f"R² Score:                     {r2:.6f}")
        
        # Create visualizations
        visualizer = AntennaVisualizer(save_dir=plot_dir)
        
        # Plot training history
        visualizer.plot_training_history(
            history,
            title=f"Enhanced Training History - {config['target_param']}",
            save_as=f"training_history_{config['target_param']}.png"
        )
        
        # Plot prediction comparison
        metrics = visualizer.plot_prediction_comparison(
            y_test_flat, y_pred_flat,
            param_name=config["target_param"],
            save_as=f"prediction_comparison_{config['target_param']}.png"
        )
        
        # Save detailed metrics
        detailed_metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'training_epochs': len(history.history['loss']),
            'best_val_loss': float(monitor.best_val_loss),
            'best_epoch': int(monitor.best_epoch)
        }
        
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(detailed_metrics, f, indent=4)
        
    else:  # multi-output
        y_test = {param: dataset['y_test'][param] for param in config["target_params"]}
        y_pred = best_model.predict(X_test, verbose=0)
        
        # If using preprocessors, inverse transform
        if config["preprocess_data"]:
            inverse_predictions = data_processor.inverse_transform_predictions(y_pred)
            y_pred = inverse_predictions
            
            # Also inverse transform y_test
            inverse_test = data_processor.inverse_transform_predictions(y_test)
            y_test = inverse_test
        
        print(f"\nMulti-Output Model Results:")
        print("=" * 80)
        
        # Calculate metrics for each parameter
        all_metrics = {}
        overall_performance = {'mse': 0, 'mae': 0, 'r2': 0, 'mape': 0}
        
        for param in config["target_params"]:
            y_test_param = y_test[param].flatten() if y_test[param].ndim > 1 else y_test[param]
            y_pred_param = y_pred[param].flatten() if y_pred[param].ndim > 1 else y_pred[param]
            
            # Calculate metrics
            mse = mean_squared_error(y_test_param, y_pred_param)
            mae = mean_absolute_error(y_test_param, y_pred_param)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_param, y_pred_param)
            mape = np.mean(np.abs((y_test_param - y_pred_param) / np.maximum(np.abs(y_test_param), 1e-8))) * 100
            
            # Calculate parameter-specific insights
            if param == 'sar':
                # SAR-specific metrics (safety critical)
                sar_violations = np.sum(y_pred_param > 1.6)  # Above safety limit
                sar_accuracy_90 = np.sum(np.abs(y_pred_param - y_test_param) < 0.16) / len(y_test_param) * 100  # Within 10% of 1.6 W/kg
                
                print(f"\n{param.upper()} (SAR - Critical for Safety):")
                print("-" * 40)
                print(f"  MSE:           {mse:.6f}")
                print(f"  RMSE:          {rmse:.6f}")
                print(f"  MAE:           {mae:.6f}")
                print(f"  MAPE:          {mape:.2f}%")
                print(f"  R²:            {r2:.6f}")
                print(f"  Safety violations: {sar_violations} samples > 1.6 W/kg")
                print(f"  High accuracy (±10%): {sar_accuracy_90:.1f}%")
                
            elif param == 'gain':
                # Gain-specific metrics (performance critical)
                gain_target_range = np.sum((y_pred_param >= 3) & (y_pred_param <= 5))  # Target range 3-5 dBi
                gain_accuracy_db = np.sum(np.abs(y_pred_param - y_test_param) < 0.5) / len(y_test_param) * 100  # Within 0.5 dB
                
                print(f"\n{param.upper()} (Antenna Gain):")
                print("-" * 40)
                print(f"  MSE:           {mse:.6f}")
                print(f"  RMSE:          {rmse:.6f}")
                print(f"  MAE:           {mae:.6f}")
                print(f"  MAPE:          {mape:.2f}%")
                print(f"  R²:            {r2:.6f}")
                print(f"  In target range (3-5 dBi): {gain_target_range} samples")
                print(f"  High accuracy (±0.5 dB): {gain_accuracy_db:.1f}%")
                
            else:
                print(f"\n{param.upper()}:")
                print("-" * 40)
                print(f"  MSE:           {mse:.6f}")
                print(f"  RMSE:          {rmse:.6f}")
                print(f"  MAE:           {mae:.6f}")
                print(f"  MAPE:          {mape:.2f}%")
                print(f"  R²:            {r2:.6f}")
            
            # Store metrics
            all_metrics[param] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'r2': float(r2)
            }
            
            # Add to overall performance (weighted average)
            weight = 2.0 if param == 'sar' else 1.5 if param == 'gain' else 1.0
            overall_performance['mse'] += mse * weight
            overall_performance['mae'] += mae * weight
            overall_performance['r2'] += r2 * weight
            overall_performance['mape'] += mape * weight
        
        # Normalize overall performance
        total_weight = sum([2.0 if p == 'sar' else 1.5 if p == 'gain' else 1.0 for p in config["target_params"]])
        for key in overall_performance:
            overall_performance[key] /= total_weight
        
        print(f"\nOVERALL MODEL PERFORMANCE:")
        print("=" * 40)
        print(f"Weighted Average MSE:  {overall_performance['mse']:.6f}")
        print(f"Weighted Average MAE:  {overall_performance['mae']:.6f}")
        print(f"Weighted Average R²:   {overall_performance['r2']:.6f}")
        print(f"Weighted Average MAPE: {overall_performance['mape']:.2f}%")
        
        # Model accuracy assessment
        if overall_performance['r2'] > 0.9:
            accuracy_grade = "EXCELLENT"
        elif overall_performance['r2'] > 0.8:
            accuracy_grade = "GOOD"
        elif overall_performance['r2'] > 0.7:
            accuracy_grade = "ACCEPTABLE"
        else:
            accuracy_grade = "NEEDS IMPROVEMENT"
            
        print(f"Model Accuracy Grade:  {accuracy_grade}")
        print(f"Prediction Confidence: {overall_performance['r2']*100:.1f}%")
        
        # Create comprehensive visualizations
        visualizer = AntennaVisualizer(save_dir=plot_dir)
        
        # Plot training history
        visualizer.plot_training_history(
            history,
            title="Enhanced Multi-Output Training History",
            save_as="training_history_multioutput.png"
        )
        
        # Plot prediction comparison for each output
        for param in config["target_params"]:
            y_test_param = y_test[param].flatten() if y_test[param].ndim > 1 else y_test[param]
            y_pred_param = y_pred[param].flatten() if y_pred[param].ndim > 1 else y_pred[param]
            
            metrics = visualizer.plot_prediction_comparison(
                y_test_param, y_pred_param,
                param_name=param,
                save_as=f"prediction_comparison_{param}.png"
            )
        
        # Create correlation matrix plot
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Prepare data for correlation analysis
            pred_data = {}
            test_data = {}
            for param in config["target_params"]:
                pred_data[f'{param}_pred'] = y_pred[param].flatten()
                test_data[f'{param}_true'] = y_test[param].flatten()
            
            # Combined correlation matrix
            combined_data = {**pred_data, **test_data}
            import pandas as pd
            df_corr = pd.DataFrame(combined_data)
            
            plt.figure(figsize=(12, 10))
            correlation_matrix = df_corr.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Prediction vs True Values Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not create correlation matrix: {e}")
        
        # Special plot for SAR if available
        if 'sar' in config["target_params"]:
            try:
                visualizer.plot_sar_distribution(
                    y_pred['sar'].flatten(),
                    threshold=1.6,  # Regulatory threshold
                    save_as="sar_distribution.png"
                )
            except Exception as e:
                print(f"Could not create SAR distribution plot: {e}")
        
        # Save comprehensive metrics
        detailed_metrics = {
            'parameters': all_metrics,
            'overall_performance': overall_performance,
            'accuracy_grade': accuracy_grade,
            'training_info': {
                'epochs': len(history.history['loss']),
                'best_val_loss': float(monitor.best_val_loss),
                'best_epoch': int(monitor.best_epoch),
                'total_parameters': int(model.count_params()),
                'trainable_parameters': int(trainable_params)
            }
        }
        
        with open(os.path.join(output_dir, "comprehensive_metrics.json"), 'w') as f:
            json.dump(detailed_metrics, f, indent=4)
    
    # Generate training report
    report_path = os.path.join(output_dir, "training_report.txt")
    with open(report_path, 'w') as f:
        f.write("ENHANCED ANTENNA PERFORMANCE PREDICTION MODEL\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Type: {config['model_type']}\n")
        f.write(f"Architecture: {config.get('model_architecture', 'enhanced')}\n")
        f.write(f"Total Parameters: {model.count_params():,}\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Validation Samples: {len(X_val)}\n")
        f.write(f"Test Samples: {len(X_test)}\n\n")
        
        if config["model_type"] == "multi_output":
            f.write(f"Overall Model Performance:\n")
            f.write(f"- Accuracy Grade: {accuracy_grade}\n")
            f.write(f"- R² Score: {overall_performance['r2']:.6f}\n")
            f.write(f"- Mean Absolute Error: {overall_performance['mae']:.6f}\n")
            f.write(f"- Prediction Confidence: {overall_performance['r2']*100:.1f}%\n\n")
            
            f.write("Parameter-specific Performance:\n")
            for param, metrics in all_metrics.items():
                f.write(f"- {param.upper()}: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}\n")
    
    print(f"\nTraining completed and results saved to {output_dir}")
    print(f"Training report: {report_path}")
    print(f"Best model: {checkpoint_path}")
    print(f"Visualizations: {plot_dir}")
    
    if config.get("use_tensorboard", True):
        print(f"TensorBoard: tensorboard --logdir {tb_log_dir}")
    
    return model, output_dir

def main():
    parser = argparse.ArgumentParser(description='Train CNN model for antenna performance prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    args = parser.parse_args()
    
    train_model(args.config)

if __name__ == '__main__':
    main() 