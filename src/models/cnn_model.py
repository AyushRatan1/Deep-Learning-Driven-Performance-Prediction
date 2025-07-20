import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np

def create_cnn_model(input_shape, output_shape, dropout_rate=0.3, learning_rate=0.001):
    """
    Create an enhanced CNN model for antenna performance prediction.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (height, width, channels)
    output_shape : int
        Number of output parameters to predict
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Initial learning rate for Adam optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled CNN model
    """
    model = models.Sequential()
    
    # Input layer and first convolutional block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                           input_shape=input_shape,
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    
    # Second convolutional block
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    
    # Third convolutional block
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    
    # Fourth convolutional block for better feature extraction
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())  # Better than Flatten for spatial invariance
    
    # Dense layers with skip connections concept
    model.add(layers.Dense(512, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(256, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(128, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate/2))
    
    # Output layer - no activation for regression (will be handled in multi-output model)
    model.add(layers.Dense(output_shape))
    
    # Compile the model with better optimizer settings
    optimizer = optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
    
    return model

def create_multioutput_model(input_shape, output_shapes, dropout_rate=0.3, learning_rate=0.001):
    """
    Create an enhanced multi-output CNN model for predicting different antenna parameters.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (height, width, channels)
    output_shapes : dict
        Dictionary with output names as keys and shapes as values
        e.g., {'s11': 1, 'gain': 1, 'sar': 1}
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Initial learning rate for Adam optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled multi-output CNN model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Shared convolutional layers with enhanced architecture
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Additional convolutional block for better feature extraction
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    
    # Global Average Pooling instead of Flatten for better generalization
    shared_features = layers.GlobalAveragePooling2D()(x)
    
    # Shared dense layers
    shared_dense = layers.Dense(512, activation='relu',
                               kernel_regularizer=regularizers.l2(0.001))(shared_features)
    shared_dense = layers.BatchNormalization()(shared_dense)
    shared_dense = layers.Dropout(dropout_rate)(shared_dense)
    
    shared_dense = layers.Dense(256, activation='relu',
                               kernel_regularizer=regularizers.l2(0.001))(shared_dense)
    shared_dense = layers.BatchNormalization()(shared_dense)
    shared_dense = layers.Dropout(dropout_rate)(shared_dense)
    
    # Specific outputs for each parameter with appropriate activations
    outputs = {}
    
    for output_name, output_size in output_shapes.items():
        # Task-specific dense layers
        task_dense = layers.Dense(128, activation='relu',
                                 kernel_regularizer=regularizers.l2(0.001))(shared_dense)
        task_dense = layers.BatchNormalization()(task_dense)
        task_dense = layers.Dropout(dropout_rate/2)(task_dense)
        
        task_dense = layers.Dense(64, activation='relu',
                                 kernel_regularizer=regularizers.l2(0.001))(task_dense)
        task_dense = layers.BatchNormalization()(task_dense)
        task_dense = layers.Dropout(dropout_rate/3)(task_dense)
        
        # Output layer with appropriate activation based on parameter type
        if output_name == 's11':
            # S11 values are typically negative (dB), use linear activation
            outputs[output_name] = layers.Dense(output_size, activation='linear',
                                              name=output_name)(task_dense)
        elif output_name == 'gain':
            # Gain can be positive or negative, use linear activation  
            outputs[output_name] = layers.Dense(output_size, activation='linear',
                                              name=output_name)(task_dense)
        elif output_name == 'sar':
            # SAR is always positive, but we're using normalized values, so linear is fine
            outputs[output_name] = layers.Dense(output_size, activation='linear',
                                              name=output_name)(task_dense)
        else:
            # Default case - linear activation for regression
            outputs[output_name] = layers.Dense(output_size, activation='linear',
                                              name=output_name)(task_dense)
    
    # Create and compile model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile with appropriate loss and metrics for each output
    losses = {}
    metrics = {}
    loss_weights = {}
    
    for output_name in output_shapes.keys():
        if output_name == 'sar':
            # SAR is critical for safety, give it higher weight
            losses[output_name] = 'mse'
            metrics[output_name] = ['mae', 'mape']
            loss_weights[output_name] = 2.0
        elif output_name == 'gain':
            # Gain is important for performance
            losses[output_name] = 'mse'
            metrics[output_name] = ['mae', 'mape']
            loss_weights[output_name] = 1.5
        else:
            # Default case
            losses[output_name] = 'mse'
            metrics[output_name] = ['mae']
            loss_weights[output_name] = 1.0
    
    optimizer = optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer, 
        loss=losses, 
        metrics=metrics,
        loss_weights=loss_weights
    )
    
    return model

def get_callbacks(checkpoint_path, patience=15, monitor='val_loss'):
    """
    Get enhanced callbacks for model training.
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to save model checkpoints
    patience : int
        Number of epochs with no improvement after which training will be stopped
    monitor : str
        Metric to monitor for callbacks
    
    Returns:
    --------
    callbacks : list
        List of callbacks for model training
    """
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='min'
        ),
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            verbose=1,
            restore_best_weights=True,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience // 3,
            verbose=1,
            min_lr=1e-7,
            mode='min'
        ),
        # Custom callback for learning rate scheduling
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.95 if epoch > 10 and epoch % 5 == 0 else lr,
            verbose=0
        )
    ]
    
    return callbacks

def create_physics_informed_model(input_shape, output_shapes, dropout_rate=0.3, learning_rate=0.001):
    """
    Create a physics-informed neural network for antenna prediction with domain knowledge.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (height, width, channels)
    output_shapes : dict
        Dictionary with output names as keys and shapes as values
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Initial learning rate for optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled physics-informed model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction with attention mechanism
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Self-attention mechanism for important pattern regions
    attention = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
    x = layers.Multiply()([x, attention])
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Physics-aware dense layers
    shared_dense = layers.Dense(512, activation='relu')(x)
    shared_dense = layers.BatchNormalization()(shared_dense)
    shared_dense = layers.Dropout(dropout_rate)(shared_dense)
    
    # Create outputs with physics constraints
    outputs = {}
    
    for output_name, output_size in output_shapes.items():
        task_dense = layers.Dense(256, activation='relu')(shared_dense)
        task_dense = layers.BatchNormalization()(task_dense)
        task_dense = layers.Dropout(dropout_rate/2)(task_dense)
        
        if output_name == 'sar':
            # SAR output with physics constraint (always positive)
            sar_raw = layers.Dense(output_size, activation='linear')(task_dense)
            outputs[output_name] = layers.Lambda(
                lambda x: tf.nn.softplus(x), name=output_name
            )(sar_raw)
        elif output_name == 'gain':
            # Gain output (can be negative or positive)
            outputs[output_name] = layers.Dense(
                output_size, activation='linear', name=output_name
            )(task_dense)
        else:
            outputs[output_name] = layers.Dense(
                output_size, activation='linear', name=output_name
            )(task_dense)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Custom loss function with physics constraints
    def physics_loss(y_true, y_pred, output_name):
        mse_loss = tf.keras.losses.mse(y_true, y_pred)
        
        if output_name == 'sar':
            # Penalty for unrealistic SAR values
            sar_penalty = tf.reduce_mean(tf.maximum(0.0, y_pred - 3.0))  # SAR > 3 W/kg penalty
            return mse_loss + 0.1 * sar_penalty
        elif output_name == 'gain':
            # Penalty for unrealistic gain values
            gain_penalty = tf.reduce_mean(tf.maximum(0.0, tf.abs(y_pred) - 15.0))  # |Gain| > 15 dBi penalty
            return mse_loss + 0.1 * gain_penalty
        else:
            return mse_loss
    
    losses = {name: lambda y_true, y_pred, n=name: physics_loss(y_true, y_pred, n) 
              for name in output_shapes.keys()}
    
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=losses, metrics=['mae'])
    
    return model 