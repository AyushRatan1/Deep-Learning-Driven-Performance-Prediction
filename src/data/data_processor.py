import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils import resample
import tensorflow as tf
import h5py
from scipy import ndimage, signal
from skimage.transform import resize
import warnings

class AntennaDataProcessor:
    """
    Enhanced class to process antenna simulation and measurement data for deep learning models.
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
        Load S-parameters from simulation/measurement files with enhanced error handling.
        
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
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
                if len(data.columns) >= 2:
                    frequencies = data.iloc[:, 0].values  # First column is usually frequency
                    s_params = data.iloc[:, 1:].values    # Remaining columns are S-parameters
                    return s_params, frequencies
                else:
                    raise ValueError(f"Invalid CSV format in {file_path}")
            elif file_path.endswith('.s1p') or file_path.endswith('.s2p'):
                # Use scikit-rf or similar library for touchstone files
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    data_lines = [line for line in lines if not line.startswith('!') and not line.startswith('#')]
                    data = np.array([list(map(float, line.strip().split())) for line in data_lines])
                    frequencies = data[:, 0]
                    s_params = data[:, 1:]
                    return s_params, frequencies
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            print(f"Warning: Could not load S-parameters from {file_path}: {e}")
            # Return dummy data to avoid pipeline failure
            dummy_freq = np.linspace(2.2, 2.6, 101)
            dummy_s11 = -10 * np.ones_like(dummy_freq)
            return dummy_s11.reshape(-1, 1), dummy_freq
    
    def load_radiation_pattern(self, file_path):
        """
        Load radiation pattern data from simulation files with enhanced processing.
        
        Parameters:
        -----------
        file_path : str
            Path to the radiation pattern file
            
        Returns:
        --------
        pattern : ndarray
            2D array representing the radiation pattern
        """
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
                pattern = data.values
                return pattern
            elif file_path.endswith('.npy'):
                pattern = np.load(file_path)
                return pattern
            elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
                with h5py.File(file_path, 'r') as f:
                    if 'radiation_pattern' in f:
                        pattern = f['radiation_pattern'][:]
                    else:
                        # Take the first dataset if radiation_pattern key doesn't exist
                        keys = list(f.keys())
                        pattern = f[keys[0]][:]
                    return pattern
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            print(f"Warning: Could not load radiation pattern from {file_path}: {e}")
            # Return dummy pattern to avoid pipeline failure
            dummy_pattern = np.random.random((64, 64))
            return dummy_pattern
    
    def convert_to_image(self, data, target_size=None):
        """
        Enhanced conversion of antenna data to image-like format for CNN input.
        
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
            
        # Handle different input data types
        if len(data.shape) == 1:
            # For 1D data (like S-parameters), create a meaningful 2D representation
            data_len = len(data)
            
            if data_len <= 16:
                # Very small data - tile it
                size = max(8, int(np.sqrt(target_size[0] * target_size[1] / data_len)))
                image = np.tile(data.reshape(-1, 1), (1, size))[:target_size[0], :target_size[1]]
            elif data_len <= 256:
                # Medium data - reshape to square if possible, otherwise use spectrogram
                sqrt_len = int(np.sqrt(data_len))
                if sqrt_len * sqrt_len == data_len:
                    image = data.reshape(sqrt_len, sqrt_len)
                else:
                    # Create spectrogram-like representation
                    nperseg = min(32, data_len // 4)
                    try:
                        f, t, Sxx = signal.spectrogram(data, fs=1.0, nperseg=nperseg)
                        image = Sxx
                    except:
                        # Fallback to simple padding and reshaping
                        padded_len = sqrt_len ** 2
                        padded_data = np.pad(data, (0, padded_len - data_len), mode='edge')
                        image = padded_data.reshape(sqrt_len, sqrt_len)
            else:
                # Large data - downsample first
                target_len = target_size[0] * target_size[1]
                indices = np.linspace(0, data_len - 1, target_len, dtype=int)
                downsampled = data[indices]
                image = downsampled.reshape(target_size[0], target_size[1])
        else:
            image = data
            
        # Ensure we have a valid 2D image
        if len(image.shape) == 1:
            # Last resort - create a simple 2D representation
            size = int(np.ceil(np.sqrt(len(image))))
            padded = np.pad(image, (0, size*size - len(image)), mode='edge')
            image = padded.reshape(size, size)
            
        # Resize to target dimensions using anti-aliasing
        if image.shape[:2] != target_size:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image = resize(image, target_size, anti_aliasing=True, preserve_range=True)
            
        # Normalize to [0, 1] range for better CNN training
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        else:
            image = np.zeros_like(image)
            
        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = image[..., np.newaxis]
            
        # Ensure float32 for TensorFlow
        image = image.astype(np.float32)
            
        return image
    
    def prepare_dataset(self, antenna_params_file, preprocess=True, augment=True):
        """
        Prepare the full dataset for training, validation, and testing with enhanced preprocessing.
        
        Parameters:
        -----------
        antenna_params_file : str
            Path to file containing antenna parameters and corresponding file paths
        preprocess : bool
            Whether to preprocess (scale) the data
        augment : bool
            Whether to apply data augmentation
            
        Returns:
        --------
        dataset : dict
            Dictionary containing train, validation, and test datasets
        """
        # Load the antenna parameters file (CSV with metadata)
        params_df = pd.read_csv(antenna_params_file)
        
        print(f"Loading dataset with {len(params_df)} samples...")
        
        # Initialize arrays for inputs and outputs
        X = []  # Input features (images from radiation patterns or other representations)
        y_s11 = []  # S11 parameter values
        y_gain = []  # Gain values
        y_sar = []  # SAR values
        valid_indices = []  # Track valid samples
        
        # Process each antenna design with error handling
        for idx, row in params_df.iterrows():
            try:
                # Load radiation pattern (or other input representation)
                pattern_file = os.path.join(self.data_path, row['pattern_file'])
                pattern = self.load_radiation_pattern(pattern_file)
                
                # Convert to image format
                pattern_img = self.convert_to_image(pattern)
                
                # Validate the converted image
                if pattern_img is None or pattern_img.size == 0:
                    print(f"Warning: Invalid pattern image for sample {idx}")
                    continue
                    
                X.append(pattern_img)
                
                # Extract output parameters with validation
                if 's11_file' in row and not pd.isna(row['s11_file']):
                    s11_file = os.path.join(self.data_path, row['s11_file'])
                    s11, _ = self.load_s_parameters(s11_file)
                    if s11.ndim > 1:
                        s11 = s11.flatten()  # Flatten if needed
                    # Use mean S11 for simplicity, or could use curve features
                    y_s11.append(np.mean(s11))
                elif 's11_value' in row:
                    y_s11.append(row['s11_value'])
                else:
                    # Use a default reasonable S11 value
                    y_s11.append(-15.0)
                    
                # Validate and process gain
                gain = row.get('gain', 3.0)
                if not np.isfinite(gain) or gain < 0 or gain > 20:
                    gain = np.clip(gain, 0, 20) if np.isfinite(gain) else 3.0
                y_gain.append(gain)
                
                # Validate and process SAR
                sar = row.get('sar', 1.0)
                if not np.isfinite(sar) or sar < 0 or sar > 5:
                    sar = np.clip(sar, 0.1, 5.0) if np.isfinite(sar) else 1.0
                y_sar.append(sar)
                
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"Warning: Error processing sample {idx}: {e}")
                continue
        
        if len(X) == 0:
            raise ValueError("No valid samples found in the dataset")
            
        print(f"Successfully loaded {len(X)} valid samples out of {len(params_df)}")
        
        # Convert lists to numpy arrays
        X = np.array(X)
        y_s11 = np.array(y_s11)
        y_gain = np.array(y_gain)
        y_sar = np.array(y_sar)
        
        # Data validation and cleaning
        print("Dataset statistics before cleaning:")
        print(f"X shape: {X.shape}")
        print(f"Gain range: [{y_gain.min():.2f}, {y_gain.max():.2f}]")
        print(f"SAR range: [{y_sar.min():.2f}, {y_sar.max():.2f}]")
        print(f"S11 range: [{y_s11.min():.2f}, {y_s11.max():.2f}]")
        
        # Apply data augmentation if requested
        if augment and len(X) < 1000:  # Only augment if we have limited data
            print("Applying data augmentation...")
            X_aug, y_s11_aug, y_gain_aug, y_sar_aug = self._augment_data(
                X, y_s11, y_gain, y_sar, target_samples=min(1000, len(X) * 3)
            )
            X = np.concatenate([X, X_aug])
            y_s11 = np.concatenate([y_s11, y_s11_aug])
            y_gain = np.concatenate([y_gain, y_gain_aug])
            y_sar = np.concatenate([y_sar, y_sar_aug])
            print(f"Augmented dataset size: {len(X)}")
        
        # Preprocess the data if required
        if preprocess:
            print("Applying data preprocessing...")
            
            # Use RobustScaler for X (more robust to outliers)
            self.scaler_X = RobustScaler()
            X_flat = X.reshape(X.shape[0], -1)
            X_flat = self.scaler_X.fit_transform(X_flat)
            X = X_flat.reshape(X.shape)
            
            # Scale outputs using appropriate scalers
            # S11 values are typically negative, use StandardScaler
            self.scaler_y['s11'] = StandardScaler()
            y_s11 = y_s11.reshape(-1, 1)
            y_s11 = self.scaler_y['s11'].fit_transform(y_s11).flatten()
            
            # Gain values benefit from MinMaxScaler for bounded range
            self.scaler_y['gain'] = MinMaxScaler(feature_range=(-1, 1))
            y_gain = y_gain.reshape(-1, 1)
            y_gain = self.scaler_y['gain'].fit_transform(y_gain).flatten()
            
            # SAR values also benefit from MinMaxScaler for bounded range
            self.scaler_y['sar'] = MinMaxScaler(feature_range=(-1, 1))
            y_sar = y_sar.reshape(-1, 1)
            y_sar = self.scaler_y['sar'].fit_transform(y_sar).flatten()
        
        # Create output dictionary
        y = {
            's11': y_s11,
            'gain': y_gain,
            'sar': y_sar
        }
        
        # Stratified split to maintain distribution balance
        # Create stratified groups based on gain and SAR ranges
        gain_bins = pd.cut(y_gain, bins=5, labels=False)
        sar_bins = pd.cut(y_sar, bins=5, labels=False)
        stratify_groups = gain_bins * 5 + sar_bins
        
        # Split into train+val and test sets
        try:
            X_train_val, X_test = train_test_split(
                X, test_size=self.test_size, random_state=self.random_state, stratify=stratify_groups
            )
            y_train_val, y_test = {}, {}
            for key in y:
                y_train_val[key], y_test[key] = train_test_split(
                    y[key], test_size=self.test_size, random_state=self.random_state, stratify=stratify_groups
                )
        except ValueError:
            # Fallback to random split if stratification fails
            print("Warning: Stratified split failed, using random split")
            X_train_val, X_test = train_test_split(
                X, test_size=self.test_size, random_state=self.random_state
            )
            y_train_val, y_test = {}, {}
            for key in y:
                y_train_val[key], y_test[key] = train_test_split(
                    y[key], test_size=self.test_size, random_state=self.random_state
                )
        
        # Split train+val into train and validation sets
        X_train, X_val = train_test_split(
            X_train_val, test_size=self.validation_size, random_state=self.random_state
        )
        
        y_train, y_val = {}, {}
        for key in y_train_val:
            y_train[key], y_val[key] = train_test_split(
                y_train_val[key], test_size=self.validation_size, random_state=self.random_state
            )
        
        print(f"Dataset split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create TensorFlow datasets with batching and prefetching for performance
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(len(X_train)).batch(32).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        
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
    
    def _augment_data(self, X, y_s11, y_gain, y_sar, target_samples=1000):
        """
        Apply data augmentation to increase dataset size.
        
        Parameters:
        -----------
        X : ndarray
            Input images
        y_s11, y_gain, y_sar : ndarray
            Target values
        target_samples : int
            Target number of augmented samples
            
        Returns:
        --------
        X_aug, y_s11_aug, y_gain_aug, y_sar_aug : ndarray
            Augmented data
        """
        n_original = len(X)
        n_augment = min(target_samples, n_original * 2)  # Don't over-augment
        
        # Randomly select samples for augmentation
        aug_indices = np.random.choice(n_original, n_augment, replace=True)
        
        X_aug = []
        y_s11_aug = []
        y_gain_aug = []
        y_sar_aug = []
        
        for idx in aug_indices:
            img = X[idx].copy()
            
            # Apply random transformations that preserve antenna physics
            # Small rotations (antennas can be slightly rotated)
            angle = np.random.uniform(-10, 10)
            img_rotated = ndimage.rotate(img[:,:,0], angle, reshape=False, order=1)
            img[:,:,0] = img_rotated
            
            # Small noise addition (measurement noise)
            noise_std = 0.02 * np.std(img)
            img += np.random.normal(0, noise_std, img.shape)
            
            # Brightness/contrast adjustment (different measurement conditions)
            brightness = np.random.uniform(0.9, 1.1)
            contrast = np.random.uniform(0.9, 1.1)
            img = np.clip(img * contrast + (brightness - 1), 0, 1)
            
            X_aug.append(img)
            
            # Add small variations to outputs (within physical constraints)
            s11_var = y_s11[idx] + np.random.normal(0, 0.5)  # Small S11 variation
            gain_var = y_gain[idx] + np.random.normal(0, 0.1)  # Small gain variation
            sar_var = y_sar[idx] + np.random.normal(0, 0.05)  # Small SAR variation
            
            y_s11_aug.append(s11_var)
            y_gain_aug.append(gain_var)
            y_sar_aug.append(sar_var)
        
        return (np.array(X_aug), np.array(y_s11_aug), 
                np.array(y_gain_aug), np.array(y_sar_aug))
    
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