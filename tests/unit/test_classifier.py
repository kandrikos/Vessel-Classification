# tests/unit/test_classifier.py
import numpy as np
import pytest
import tensorflow as tf

from src.models.classifier import VesselClassifier


class TestVesselClassifier:
    """Test suite for the VesselClassifier class"""

    @pytest.fixture
    def model_params(self):
        """Fixture for model parameters"""
        return {
            "initial_filters": 32,  # Smaller for faster tests
            "initial_kernel_size": 5,
            "initial_stride": 2,
            "block_filters": [32, 64],  # Reduced depth for testing
            "block_kernel_size": 3,
            "dropout_rate": 0.3,
            "l2_reg": 0.01,
            "use_augmentation": False,
        }

    @pytest.fixture
    def classifier(self, model_params):
        """Fixture for a classifier instance with test configuration"""
        input_shape = (95, 126, 3)
        num_classes = 5
        return VesselClassifier(input_shape, num_classes, model_params)

    def test_model_creation(self, classifier):
        """Test if the model is created with correct architecture"""
        # Check basic properties
        assert classifier.model is not None
        assert isinstance(classifier.model, tf.keras.Model)

        # Check input shape
        assert classifier.model.input_shape == (None, 95, 126, 3)

        # Check output shape
        assert classifier.model.output_shape == (None, 5)

    def test_model_compilation(self, classifier):
        """Test model compilation with different optimizers"""
        # Test SGD optimizer
        sgd_params = {"optimizer": "sgd", "learning_rate": 0.001, "momentum": 0.9}
        classifier.compile_model(sgd_params)
        assert classifier.model.optimizer.__class__.__name__ == "SGD"

        # Test Adam optimizer
        adam_params = {"optimizer": "adam", "learning_rate": 0.0005}
        classifier.compile_model(adam_params)
        assert classifier.model.optimizer.__class__.__name__ == "Adam"

        # Check metrics
        metric_names = [m.name for m in classifier.model.metrics]
        assert "accuracy" in metric_names
        assert "precision" in metric_names
        assert "recall" in metric_names

    def test_prediction_shape(self, classifier):
        """Test if predictions have correct shape"""
        # Compile model first
        classifier.compile_model()

        # Create dummy input data
        batch_size = 4
        dummy_input = np.random.random((batch_size, 95, 126, 3))

        # Get predictions
        predictions = classifier.predict(dummy_input)

        # Check shape and properties
        assert predictions.shape == (batch_size, 5)
        assert np.allclose(
            np.sum(predictions, axis=1), 1.0
        )  # Sum of probabilities should be 1
        assert np.all(predictions >= 0) and np.all(
            predictions <= 1
        )  # Probabilities between 0 and 1
