import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import h5py

class AntennaDataProcessor:
    """
    Class to process antenna simulation and measurement data for deep learning models.
    """
    
    def __init__(self, data_path, target_size=(128, 128), test_size=0.2, validation_size=0.1, random_state=42):
        """
        Initialize the data processor.
        
        Parameters:
        -----------
        data_path : str
            Path to the raw data directory
        target_size : tuple
            Target size for input images (height, width)
        test_size : float
            Proportion of data to use for testing
        validation_size : float
            Proportion of training data to use for validation
        random_state : int
            Random seed for reproducibility
        """
        self.data_path = data_path
        self.target_size = target_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.scaler_X = None
        self.scaler_y = {}
        
    def load_s_parameters(self, file_path):
        """
        Load S-parameters from simulation/measurement files.
        
        Parameters:
        -----------
        file_path : str
            Path to the S-parameter file
            
        Returns:
        --------
        s_params : ndarray
            Array of S-parameters
        frequencies : ndarray
            Array of frequencies
        """
        # This is a placeholder. Actual implementation would depend on the file format
        # Common formats include .s1p, .s2p, or CSV files
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
            frequencies = data.iloc[:, 0].values  # First column is usually frequency
            s_params = data.iloc[:, 1:].values    # Remaining columns are S-parameters
            return s_params, frequencies
        elif file_path.endswith('.s1p') or file_path.endswith('.s2p'):
            # Use scikit-rf or similar library for touchstone files
            # This is a simplified placeholder
            with open(file_path, 'r') as f:
                lines = f.readlines()
                data_lines = [line for line in lines if not line.startswith('!') and not line.startswith('#')]
                data = np.array([list(map(float, line.strip().split())) for line in data_lines])
                frequencies = data[:, 0]
                s_params = data[:, 1:]
                return s_params, frequencies
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def load_radiation_pattern(self, file_path):
        """
        Load radiation pattern data from simulation files.
        
        Parameters:
        -----------
        file_path : str
            Path to the radiation pattern file
            
        Returns:
        --------
        pattern : ndarray
            2D array representing the radiation pattern
        """
        # Placeholder implementation - replace with actual file format handling
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
            # Assuming the CSV contains a pattern matrix or values that can be reshaped
            pattern = data.values
            return pattern
        elif file_path.endswith('.npy'):
            pattern = np.load(file_path)
            return pattern
        elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            with h5py.File(file_path, 'r') as f:
                pattern = f['radiation_pattern'][:]
                return pattern
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def convert_to_image(self, data, target_size=None):
        """
        Convert antenna data to image-like format for CNN input.
        
        Parameters:
        -----------
        data : ndarray
            Input data array
        target_size : tuple, optional
            Target size for the output image
            
        Returns:
        --------
        image : ndarray
            Data converted to image format of shape target_size + (channels,)
        """
        if target_size is None:
            target_size = self.target_size
            
        # If data is 1D (like S-parameters), convert to 2D representation
        if len(data.shape) == 1:
            # Option 1: Simple reshaping for frequency-domain data
            size = int(np.sqrt(data.shape[0]))
            if size * size == data.shape[0]:
                image = data.reshape(size, size)
            else:
                # Option 2: Create a spectrogram-like representation
                from scipy import signal
                f, t, Sxx = signal.spectrogram(data, fs=1.0, nperseg=min(256, len(data)//2))
                image = Sxx
        else:
            image = data
            
        # Resize to target dimensions
        if image.shape[:2] != target_size:
            from skimage.transform import resize
            image = resize(image, target_size, anti_aliasing=True)
            
        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = image[..., np.newaxis]
            
        return image
    
    def prepare_dataset(self, antenna_params_file, preprocess=True):
        """
        Prepare the full dataset for training, validation, and testing.
        
        Parameters:
        -----------
        antenna_params_file : str
            Path to file containing antenna parameters and corresponding file paths
        preprocess : bool
            Whether to preprocess (scale) the data
            
        Returns:
        --------
        dataset : dict
            Dictionary containing train, validation, and test datasets
        """
        # Load the antenna parameters file (CSV with metadata)
        params_df = pd.read_csv(antenna_params_file)
        
        # Initialize arrays for inputs and outputs
        X = []  # Input features (images from radiation patterns or other representations)
        y_s11 = []  # S11 parameter values
        y_gain = []  # Gain values
        y_sar = []  # SAR values
        
        # Process each antenna design
        for idx, row in params_df.iterrows():
            # Load radiation pattern (or other input representation)
            pattern_file = os.path.join(self.data_path, row['pattern_file'])
            pattern = self.load_radiation_pattern(pattern_file)
            
            # Convert to image format
            pattern_img = self.convert_to_image(pattern)
            X.append(pattern_img)
            
            # Extract output parameters
            if 's11_file' in row:
                s11_file = os.path.join(self.data_path, row['s11_file'])
                s11, _ = self.load_s_parameters(s11_file)
                y_s11.append(s11.flatten())  # Flatten if needed
            else:
                y_s11.append(row['s11_value'])
                
            y_gain.append(row['gain'])
            y_sar.append(row['sar'])
        
        # Convert lists to numpy arrays
        X = np.array(X)
        y_s11 = np.array(y_s11)
        y_gain = np.array(y_gain)
        y_sar = np.array(y_sar)
        
        # Preprocess the data if required
        if preprocess:
            self.scaler_X = StandardScaler()
            X_flat = X.reshape(X.shape[0], -1)
            X_flat = self.scaler_X.fit_transform(X_flat)
            X = X_flat.reshape(X.shape)
            
            # Scale outputs
            self.scaler_y['s11'] = StandardScaler()
            if len(y_s11.shape) == 1:
                y_s11 = y_s11.reshape(-1, 1)
                y_s11 = self.scaler_y['s11'].fit_transform(y_s11)
            else:
                y_s11_flat = y_s11.reshape(y_s11.shape[0], -1)
                y_s11_flat = self.scaler_y['s11'].fit_transform(y_s11_flat)
                y_s11 = y_s11_flat.reshape(y_s11.shape)
            
            self.scaler_y['gain'] = StandardScaler()
            y_gain = y_gain.reshape(-1, 1)
            y_gain = self.scaler_y['gain'].fit_transform(y_gain)
            
            self.scaler_y['sar'] = StandardScaler()
            y_sar = y_sar.reshape(-1, 1)
            y_sar = self.scaler_y['sar'].fit_transform(y_sar)
        
        # Create output dictionary
        y = {
            's11': y_s11,
            'gain': y_gain,
            'sar': y_sar
        }
        
        # Split into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = {}, {}, {}, {}
        X_train_val, X_test = train_test_split(
            X, test_size=self.test_size, random_state=self.random_state
        )
        
        for key in y:
            y_train_val[key], y_test[key] = train_test_split(
                y[key], test_size=self.test_size, random_state=self.random_state
            )
        
        # Split train+val into train and validation sets
        X_train, X_val, y_train, y_val = {}, {}, {}, {}
        X_train, X_val = train_test_split(
            X_train_val, test_size=self.validation_size, random_state=self.random_state
        )
        
        for key in y_train_val:
            y_train[key], y_val[key] = train_test_split(
                y_train_val[key], test_size=self.validation_size, random_state=self.random_state
            )
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        
        return {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def inverse_transform_predictions(self, predictions):
        """
        Inverse transform the scaled predictions back to original scale.
        
        Parameters:
        -----------
        predictions : dict
            Dictionary of model predictions for each output
            
        Returns:
        --------
        inverse_predictions : dict
            Dictionary of inverse-transformed predictions
        """
        inverse_predictions = {}
        
        for key, pred in predictions.items():
            if key in self.scaler_y:
                if len(pred.shape) == 1:
                    pred = pred.reshape(-1, 1)
                    inverse_predictions[key] = self.scaler_y[key].inverse_transform(pred)
                else:
                    pred_flat = pred.reshape(pred.shape[0], -1)
                    inverse_pred_flat = self.scaler_y[key].inverse_transform(pred_flat)
                    inverse_predictions[key] = inverse_pred_flat.reshape(pred.shape)
            else:
                inverse_predictions[key] = pred
                
        return inverse_predictions
    
    def save_preprocessors(self, save_path):
        """
        Save the preprocessors for later use.
        
        Parameters:
        -----------
        save_path : str
            Path to save the preprocessors
        """
        import joblib
        
        os.makedirs(save_path, exist_ok=True)
        
        if self.scaler_X is not None:
            joblib.dump(self.scaler_X, os.path.join(save_path, 'scaler_X.joblib'))
            
        for key, scaler in self.scaler_y.items():
            joblib.dump(scaler, os.path.join(save_path, f'scaler_y_{key}.joblib'))
    
    def load_preprocessors(self, load_path):
        """
        Load the preprocessors from saved files.
        
        Parameters:
        -----------
        load_path : str
            Path to load the preprocessors from
        """
        import joblib
        
        scaler_X_path = os.path.join(load_path, 'scaler_X.joblib')
        if os.path.exists(scaler_X_path):
            self.scaler_X = joblib.load(scaler_X_path)
            
        for key in ['s11', 'gain', 'sar']:
            scaler_y_path = os.path.join(load_path, f'scaler_y_{key}.joblib')
            if os.path.exists(scaler_y_path):
                self.scaler_y[key] = joblib.load(scaler_y_path) 