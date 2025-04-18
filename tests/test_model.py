import os
import sys
import unittest
import numpy as np
import tensorflow as tf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn_model import create_cnn_model, create_multioutput_model

class TestModelArchitecture(unittest.TestCase):
    """
    Test cases for CNN model architecture.
    """
    
    def test_single_output_model(self):
        """Test the single output model creation and shapes."""
        # Create model with sample input and output shapes
        input_shape = (128, 128, 1)
        output_shape = 1
        
        model = create_cnn_model(
            input_shape=input_shape,
            output_shape=output_shape,
            dropout_rate=0.3,
            learning_rate=0.001
        )
        
        # Check model type
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 128, 128, 1))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 1))
        
        # Test with a sample input
        test_input = np.random.random((2, 128, 128, 1))
        test_output = model.predict(test_input)
        
        # Check output shape with sample input
        self.assertEqual(test_output.shape, (2, 1))
    
    def test_multi_output_model(self):
        """Test the multi-output model creation and shapes."""
        # Create model with sample input and output shapes
        input_shape = (128, 128, 1)
        output_shapes = {
            's11': 101,
            'gain': 1,
            'sar': 1
        }
        
        model = create_multioutput_model(
            input_shape=input_shape,
            output_shapes=output_shapes,
            dropout_rate=0.3,
            learning_rate=0.001
        )
        
        # Check model type
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 128, 128, 1))
        
        # Check output shapes
        self.assertEqual(model.output_shape['s11'], (None, 101))
        self.assertEqual(model.output_shape['gain'], (None, 1))
        self.assertEqual(model.output_shape['sar'], (None, 1))
        
        # Test with a sample input
        test_input = np.random.random((2, 128, 128, 1))
        test_output = model.predict(test_input)
        
        # Check output shapes with sample input
        self.assertEqual(test_output['s11'].shape, (2, 101))
        self.assertEqual(test_output['gain'].shape, (2, 1))
        self.assertEqual(test_output['sar'].shape, (2, 1))

if __name__ == '__main__':
    unittest.main() 