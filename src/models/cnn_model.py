import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def create_cnn_model(input_shape, output_shape, dropout_rate=0.3, learning_rate=0.001):
    """
    Create a CNN model for antenna performance prediction.
    
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
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    
    # Second convolutional block
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    
    # Third convolutional block
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    
    # Flatten and fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(output_shape))
    
    # Compile the model
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def get_callbacks(checkpoint_path, patience=10):
    """
    Get callbacks for model training.
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to save model checkpoints
    patience : int
        Number of epochs with no improvement after which training will be stopped
    
    Returns:
    --------
    callbacks : list
        List of callbacks for model training
    """
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 3,
            verbose=1,
            min_lr=1e-6
        )
    ]
    return callbacks

def create_multioutput_model(input_shape, output_shapes, dropout_rate=0.3, learning_rate=0.001):
    """
    Create a multi-output CNN model for predicting different antenna parameters.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (height, width, channels)
    output_shapes : dict
        Dictionary with output names as keys and shapes as values
        e.g., {'s11': 101, 'gain': 2, 'sar': 1}
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
    
    # Shared layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Flatten shared features
    shared_features = layers.Flatten()(x)
    shared_dense = layers.Dense(256, activation='relu')(shared_features)
    shared_dense = layers.BatchNormalization()(shared_dense)
    shared_dense = layers.Dropout(dropout_rate)(shared_dense)
    
    # Specific outputs for each parameter
    outputs = {}
    for output_name, output_size in output_shapes.items():
        output_dense = layers.Dense(128, activation='relu')(shared_dense)
        output_dense = layers.BatchNormalization()(output_dense)
        output_dense = layers.Dropout(dropout_rate/2)(output_dense)
        outputs[output_name] = layers.Dense(output_size, name=output_name)(output_dense)
    
    # Create and compile model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile with appropriate loss and metrics for each output
    losses = {output_name: 'mse' for output_name in output_shapes.keys()}
    metrics = {output_name: 'mae' for output_name in output_shapes.keys()}
    
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
    
    return model 