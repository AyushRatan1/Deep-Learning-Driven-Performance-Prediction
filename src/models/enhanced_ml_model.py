import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

class EnhancedAntennaPredictor:
    """
    Enhanced Machine Learning model for high-accuracy antenna performance prediction.
    Uses ensemble methods combining XGBoost, LightGBM, CatBoost, and Random Forest.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.ensemble = None
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.feature_names = None
        self.target_scalers = {}
        
    def create_features(self, data):
        """
        Create advanced feature engineering for antenna parameters.
        
        Parameters:
        -----------
        data : dict
            Dictionary containing antenna parameters
            
        Returns:
        --------
        features : ndarray
            Engineered feature array
        """
        features = []
        
        # Basic parameters
        substrate_thickness = data['substrate_thickness']
        substrate_permittivity = data['substrate_permittivity']
        patch_width = data['patch_width']
        patch_length = data['patch_length']
        feed_position = data['feed_position']
        bending_radius = data['bending_radius']
        
        # Basic features
        features.extend([
            substrate_thickness,
            substrate_permittivity,
            patch_width,
            patch_length,
            feed_position,
            bending_radius
        ])
        
        # Physics-based derived features
        # 1. Antenna area and aspect ratio
        area = patch_width * patch_length
        aspect_ratio = patch_width / patch_length if patch_length > 0 else 1.0
        
        # 2. Electrical properties
        wavelength_in_dielectric = 125 / np.sqrt(substrate_permittivity)  # ~2.4GHz
        patch_width_normalized = patch_width / wavelength_in_dielectric
        patch_length_normalized = patch_length / wavelength_in_dielectric
        
        # 3. Feed position metrics
        feed_offset_width = abs(feed_position - patch_width/2)
        feed_offset_length = abs(feed_position - patch_length/2)
        
        # 4. Bending effects
        bending_factor = 1.0 / (1.0 + bending_radius/10.0)  # Normalized bending effect
        
        # 5. Substrate effects
        substrate_ratio = substrate_thickness / substrate_permittivity
        dielectric_loading = substrate_permittivity * substrate_thickness
        
        # Add derived features
        features.extend([
            area,
            aspect_ratio,
            patch_width_normalized,
            patch_length_normalized,
            feed_offset_width,
            feed_offset_length,
            bending_factor,
            substrate_ratio,
            dielectric_loading,
            # Cross-products for interaction effects
            substrate_thickness * substrate_permittivity,
            patch_width * patch_length,
            bending_radius * substrate_thickness,
            feed_position * patch_width,
            # Higher-order features
            np.sqrt(area),
            np.log1p(bending_radius),
            substrate_permittivity ** 0.5,
            # Resonance-related features
            wavelength_in_dielectric / 2.0,  # Half wavelength
            (patch_width + patch_length) / 2.0,  # Average dimension
        ])
        
        self.feature_names = [
            'substrate_thickness', 'substrate_permittivity', 'patch_width', 'patch_length',
            'feed_position', 'bending_radius', 'area', 'aspect_ratio',
            'patch_width_norm', 'patch_length_norm', 'feed_offset_w', 'feed_offset_l',
            'bending_factor', 'substrate_ratio', 'dielectric_loading',
            'thickness_x_permittivity', 'width_x_length', 'bending_x_thickness',
            'feed_x_width', 'sqrt_area', 'log_bending', 'sqrt_permittivity',
            'half_wavelength', 'avg_dimension'
        ]
        
        return np.array(features)
    
    def prepare_data(self, params_file):
        """
        Load and prepare data for training with advanced feature engineering.
        
        Parameters:
        -----------
        params_file : str
            Path to antenna parameters CSV file
            
        Returns:
        --------
        X : ndarray
            Feature matrix
        y : dict
            Target values for each parameter
        """
        # Load data
        df = pd.read_csv(params_file)
        print(f"Loaded {len(df)} samples from {params_file}")
        
        # Create features
        X = []
        for idx, row in df.iterrows():
            features = self.create_features(row.to_dict())
            X.append(features)
        
        X = np.array(X)
        
        # Prepare targets
        y = {
            'gain': df['gain'].values,
            'sar': df['sar'].values,
            's11': df.get('s11_value', pd.Series(np.full(len(df), -15.0))).values
        }
        
        # Data validation and cleaning
        for param, values in y.items():
            # Remove outliers using IQR method
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            outliers = (values < lower_bound) | (values > upper_bound)
            if np.sum(outliers) > 0:
                print(f"Found {np.sum(outliers)} outliers in {param}, clipping to bounds")
                y[param] = np.clip(values, lower_bound, upper_bound)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Feature names: {len(self.feature_names)}")
        
        return X, y
    
    def train(self, params_file, target_params=['gain', 'sar', 's11']):
        """
        Train the enhanced ensemble model.
        
        Parameters:
        -----------
        params_file : str
            Path to training data
        target_params : list
            List of target parameters to predict
        """
        print("="*80)
        print("ENHANCED ANTENNA PERFORMANCE PREDICTION TRAINING")
        print("="*80)
        
        # Prepare data
        X, y = self.prepare_data(params_file)
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        print("\nPerforming feature selection...")
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(15, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X_scaled, y['gain'])  # Use gain for selection
        
        selected_features = self.feature_selector.get_support()
        selected_feature_names = [name for i, name in enumerate(self.feature_names) if selected_features[i]]
        print(f"Selected {len(selected_feature_names)} features: {selected_feature_names}")
        
        # Split data
        indices = np.arange(len(X_selected))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=self.random_state
        )
        
        X_train = X_selected[train_idx]
        X_test = X_selected[test_idx]
        
        y_train_dict = {k: v[train_idx] for k, v in y.items()}
        y_test_dict = {k: v[test_idx] for k, v in y.items()}
        
        # Train models for each target parameter
        for param in target_params:
            if param not in y:
                print(f"Warning: {param} not found in data, skipping")
                continue
                
            print(f"\n{'='*50}")
            print(f"Training models for {param.upper()}")
            print(f"{'='*50}")
            
            y_train = y_train_dict[param]
            y_test = y_test_dict[param]
            
            # Scale target values
            self.target_scalers[param] = RobustScaler()
            y_train_scaled = self.target_scalers[param].fit_transform(y_train.reshape(-1, 1)).flatten()
            
            # Train individual models
            models_for_param = {}
            
            # 1. XGBoost
            print("Training XGBoost...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1
            )
            xgb_model.fit(X_train, y_train_scaled)
            models_for_param['xgb'] = xgb_model
            
            # 2. LightGBM
            print("Training LightGBM...")
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )
            lgb_model.fit(X_train, y_train_scaled)
            models_for_param['lgb'] = lgb_model
            
            # 3. CatBoost (skip if all targets are the same)
            if len(np.unique(y_train_scaled)) > 1:
                print("Training CatBoost...")
                cb_model = cb.CatBoostRegressor(
                    iterations=200,
                    depth=6,
                    learning_rate=0.1,
                    random_seed=self.random_state,
                    verbose=False
                )
                cb_model.fit(X_train, y_train_scaled)
                models_for_param['catboost'] = cb_model
            else:
                print("Skipping CatBoost (all targets are identical)")
                models_for_param['catboost'] = None
            
            # 4. Random Forest
            print("Training Random Forest...")
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train_scaled)
            models_for_param['rf'] = rf_model
            
            # Create ensemble
            print("Creating ensemble...")
            ensemble_models = [
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('rf', rf_model)
            ]
            if models_for_param['catboost'] is not None:
                ensemble_models.append(('catboost', models_for_param['catboost']))
            
            ensemble = VotingRegressor(ensemble_models)
            ensemble.fit(X_train, y_train_scaled)
            models_for_param['ensemble'] = ensemble
            
            self.models[param] = models_for_param
            
            # Evaluate on test set
            y_pred_scaled = ensemble.predict(X_test)
            y_pred = self.target_scalers[param].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8))) * 100
            
            print(f"\n{param.upper()} Results:")
            print("-" * 30)
            print(f"MSE:  {mse:.6f}")
            print(f"MAE:  {mae:.6f}")
            print(f"R²:   {r2:.6f}")
            print(f"MAPE: {mape:.2f}%")
            
            # Feature importance
            if hasattr(ensemble.estimators_[0], 'feature_importances_'):
                importances = ensemble.estimators_[0].feature_importances_
                feature_importance = list(zip(selected_feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                print(f"\nTop 5 Important Features for {param}:")
                for feat, imp in feature_importance[:5]:
                    print(f"  {feat}: {imp:.4f}")
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        
    def predict(self, antenna_params):
        """
        Make predictions for given antenna parameters.
        
        Parameters:
        -----------
        antenna_params : dict
            Dictionary containing antenna parameters
            
        Returns:
        --------
        predictions : dict
            Predicted values for each parameter
        """
        # Create features
        features = self.create_features(antenna_params)
        X = features.reshape(1, -1)
        
        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        predictions = {}
        for param, models in self.models.items():
            # Use ensemble for prediction
            ensemble = models['ensemble']
            pred_scaled = ensemble.predict(X_selected)
            pred = self.target_scalers[param].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            predictions[param] = pred
            
        return predictions
    
    def save_model(self, filepath):
        """Save the trained model."""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'target_scalers': self.target_scalers,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.target_scalers = model_data['target_scalers']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")

def main():
    """Main function for training the enhanced model."""
    predictor = EnhancedAntennaPredictor()
    
    # Train the model
    predictor.train('data/raw/antenna_params.csv')
    
    # Save the model
    os.makedirs('models/enhanced_ml', exist_ok=True)
    predictor.save_model('models/enhanced_ml/antenna_predictor.joblib')
    
    # Test prediction
    test_params = {
        'substrate_thickness': 1.5,
        'substrate_permittivity': 2.5,
        'patch_width': 35.0,
        'patch_length': 40.0,
        'feed_position': 10.0,
        'bending_radius': 15.0
    }
    
    predictions = predictor.predict(test_params)
    print(f"\nTest Prediction:")
    for param, value in predictions.items():
        print(f"  {param}: {value:.4f}")

if __name__ == "__main__":
    main() 