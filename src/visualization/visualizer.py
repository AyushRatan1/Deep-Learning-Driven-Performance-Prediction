import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os

class AntennaVisualizer:
    """
    Class for visualizing antenna performance parameters and model results.
    """
    
    def __init__(self, save_dir='./plots'):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        save_dir : str
            Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set default plot style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_s_parameters(self, frequencies, s_params, labels=None, title='S-Parameters', save_as=None):
        """
        Plot S-parameters vs frequency.
        
        Parameters:
        -----------
        frequencies : ndarray
            Array of frequency values
        s_params : ndarray or list of ndarrays
            S-parameter values, can be multiple sets for comparison
        labels : list of str, optional
            Labels for each set of S-parameters
        title : str
            Plot title
        save_as : str, optional
            Filename to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        if isinstance(s_params, list):
            for i, s_param in enumerate(s_params):
                label = labels[i] if labels is not None and i < len(labels) else f'Set {i+1}'
                plt.plot(frequencies, s_param, label=label)
        else:
            plt.plot(frequencies, s_params)
            
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('S11 (dB)')
        plt.title(title)
        plt.grid(True)
        
        if labels is not None:
            plt.legend()
            
        if save_as:
            plt.savefig(os.path.join(self.save_dir, save_as), dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def plot_radiation_pattern(self, pattern, title='Radiation Pattern', is_3d=False, save_as=None):
        """
        Plot antenna radiation pattern.
        
        Parameters:
        -----------
        pattern : ndarray
            2D array of radiation pattern values
        title : str
            Plot title
        is_3d : bool
            Whether to show a 3D plot
        save_as : str, optional
            Filename to save the plot
        """
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Create theta and phi grids
            theta = np.linspace(0, 2*np.pi, pattern.shape[0])
            phi = np.linspace(0, np.pi, pattern.shape[1])
            
            THETA, PHI = np.meshgrid(theta, phi)
            
            # Convert to Cartesian coordinates
            R = pattern
            X = R * np.sin(PHI) * np.cos(THETA)
            Y = R * np.sin(PHI) * np.sin(THETA)
            Z = R * np.cos(PHI)
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the 3D surface
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True)
            
            ax.set_title(title)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            
        else:
            plt.figure(figsize=(10, 8))
            contour = plt.contourf(pattern, cmap='viridis', levels=50)
            plt.colorbar(contour)
            plt.title(title)
            plt.xlabel('Azimuth Angle (°)')
            plt.ylabel('Elevation Angle (°)')
            
        if save_as:
            plt.savefig(os.path.join(self.save_dir, save_as), dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def plot_prediction_comparison(self, y_true, y_pred, param_name, title=None, save_as=None):
        """
        Plot comparison between true and predicted values.
        
        Parameters:
        -----------
        y_true : ndarray
            True values
        y_pred : ndarray
            Predicted values
        param_name : str
            Name of the parameter being compared
        title : str, optional
            Plot title
        save_as : str, optional
            Filename to save the plot
        """
        if title is None:
            title = f'{param_name} - True vs Predicted'
            
        # Flatten arrays if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            # If we're dealing with a spectrum or multiple values, plot differently
            # Plot example comparisons
            plt.figure(figsize=(14, 10))
            
            for i in range(min(5, y_true.shape[0])):
                plt.subplot(5, 1, i+1)
                plt.plot(y_true[i], label='True', color='blue')
                plt.plot(y_pred[i], label='Predicted', color='red', linestyle='--')
                plt.legend()
                plt.title(f'Sample {i+1}')
                
            plt.tight_layout()
        else:
            # If single values, use scatter plot
            if len(y_true.shape) > 1:
                y_true = y_true.ravel()
            if len(y_pred.shape) > 1:
                y_pred = y_pred.ravel()
                
            plt.figure(figsize=(10, 8))
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Create scatter plot
            plt.scatter(y_true, y_pred, alpha=0.6)
            
            # Add perfect prediction line
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect prediction')
            
            plt.xlabel(f'True {param_name}')
            plt.ylabel(f'Predicted {param_name}')
            plt.title(f'{title}\nMSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
            plt.legend()
            plt.grid(True)
            
        if save_as:
            plt.savefig(os.path.join(self.save_dir, save_as), dpi=300, bbox_inches='tight')
            
        plt.show()
        
        return {'mse': mse, 'mae': mae, 'r2': r2}
    
    def plot_sar_distribution(self, sar_values, threshold=1.6, title='SAR Distribution', save_as=None):
        """
        Plot distribution of SAR values with regulatory threshold.
        
        Parameters:
        -----------
        sar_values : ndarray
            Array of SAR values
        threshold : float
            Regulatory threshold (typically 1.6 W/kg)
        title : str
            Plot title
        save_as : str, optional
            Filename to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        sns.histplot(sar_values, kde=True)
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold} W/kg')
        
        # Calculate percentage of designs exceeding threshold
        exceed_pct = np.mean(sar_values > threshold) * 100
        
        plt.xlabel('SAR (W/kg)')
        plt.ylabel('Frequency')
        plt.title(f'{title}\n{exceed_pct:.1f}% of designs exceed threshold')
        plt.legend()
        
        if save_as:
            plt.savefig(os.path.join(self.save_dir, save_as), dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def plot_bending_effect(self, bending_radii, performance_params, param_name, title=None, save_as=None):
        """
        Plot effect of bending on antenna performance.
        
        Parameters:
        -----------
        bending_radii : list or ndarray
            Array of bending radii values
        performance_params : ndarray
            Corresponding performance parameter values
        param_name : str
            Name of the performance parameter
        title : str, optional
            Plot title
        save_as : str, optional
            Filename to save the plot
        """
        if title is None:
            title = f'Effect of Bending on {param_name}'
            
        plt.figure(figsize=(10, 6))
        
        plt.plot(bending_radii, performance_params, 'o-')
        plt.axvline(x=10, color='r', linestyle='--', label='Comfort threshold: 10 mm')
        
        plt.xlabel('Bending Radius (mm)')
        plt.ylabel(param_name)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        
        if save_as:
            plt.savefig(os.path.join(self.save_dir, save_as), dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def plot_training_history(self, history, metrics=None, title='Training History', save_as=None):
        """
        Plot training history for the model.
        
        Parameters:
        -----------
        history : tf.keras.callbacks.History
            Training history from model.fit()
        metrics : list of str, optional
            Metrics to plot (defaults to all)
        title : str
            Plot title
        save_as : str, optional
            Filename to save the plot
        """
        if metrics is None:
            # Find all available metrics
            metrics = []
            for key in history.history.keys():
                if not key.startswith('val_'):
                    metrics.append(key)
        
        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(15, 5))
        
        if num_metrics == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            axes[i].plot(history.history[metric], label=f'Training {metric}')
            if f'val_{metric}' in history.history:
                axes[i].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            
            axes[i].set_title(f'{metric} over epochs')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric)
            axes[i].legend()
            axes[i].grid(True)
            
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_as:
            plt.savefig(os.path.join(self.save_dir, save_as), dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def plot_feature_importance(self, features, importances, title='Feature Importance', save_as=None):
        """
        Plot feature importance for model interpretability.
        
        Parameters:
        -----------
        features : list of str
            Feature names
        importances : ndarray
            Corresponding importance values
        title : str
            Plot title
        save_as : str, optional
            Filename to save the plot
        """
        # Sort features by importance
        indices = np.argsort(importances)
        features = [features[i] for i in indices]
        importances = importances[indices]
        
        plt.figure(figsize=(10, 8))
        
        plt.barh(range(len(features)), importances, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(title)
        
        if save_as:
            plt.savefig(os.path.join(self.save_dir, save_as), dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def plot_correlation_matrix(self, data, labels=None, title='Correlation Matrix', save_as=None):
        """
        Plot correlation matrix for antenna design parameters.
        
        Parameters:
        -----------
        data : ndarray or DataFrame
            Data matrix
        labels : list of str, optional
            Column labels
        title : str
            Plot title
        save_as : str, optional
            Filename to save the plot
        """
        if isinstance(data, np.ndarray):
            if labels is None:
                labels = [f'Param_{i+1}' for i in range(data.shape[1])]
            data = pd.DataFrame(data, columns=labels)
            
        corr = data.corr()
        
        plt.figure(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt='.2f')
        
        plt.title(title)
        
        if save_as:
            plt.savefig(os.path.join(self.save_dir, save_as), dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def plot_parameter_sweep(self, sweep_param, sweep_values, performance_metrics, 
                            metric_name='Gain', title=None, save_as=None):
        """
        Plot effect of parameter sweep on antenna performance.
        
        Parameters:
        -----------
        sweep_param : str
            Parameter being swept
        sweep_values : list or ndarray
            Values of the swept parameter
        performance_metrics : list or ndarray
            Corresponding performance metrics
        metric_name : str
            Name of the performance metric
        title : str, optional
            Plot title
        save_as : str, optional
            Filename to save the plot
        """
        if title is None:
            title = f'Effect of {sweep_param} on {metric_name}'
            
        plt.figure(figsize=(10, 6))
        
        plt.plot(sweep_values, performance_metrics, 'o-')
        
        plt.xlabel(sweep_param)
        plt.ylabel(metric_name)
        plt.title(title)
        plt.grid(True)
        
        # Annotate best value
        best_idx = np.argmax(performance_metrics) if metric_name in ['Gain', 'Efficiency'] else np.argmin(performance_metrics)
        best_value = sweep_values[best_idx]
        best_metric = performance_metrics[best_idx]
        
        plt.annotate(f'Best: {best_metric:.3f} @ {best_value}',
                    xy=(best_value, best_metric),
                    xytext=(0, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->'),
                    ha='center')
        
        if save_as:
            plt.savefig(os.path.join(self.save_dir, save_as), dpi=300, bbox_inches='tight')
            
        plt.show() 