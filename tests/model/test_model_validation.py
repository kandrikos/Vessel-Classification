import json
import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from src.models.classifier import VesselClassifier
from src.data.data_generator import VesselDataGenerator


class TestModelValidation:
    """Tests for validating model performance and quality"""

    @pytest.fixture
    def trained_model(self, tmp_path):
        """Fixture that returns a path to a trained model for testing"""
        # For real tests, you might want to download a pre-trained model
        # Here we'll create a dummy model for testing
        input_shape = (95, 126, 3)
        model = VesselClassifier(input_shape, num_classes=5).model

        # Save model
        model_path = tmp_path / "test_model.keras"
        model.save(model_path)

        return model_path

    @pytest.fixture
    def benchmark_metrics(self):
        """Benchmark metrics that the model should meet"""
        return {
            "accuracy": 0.95,  # 95% accuracy minimum
            "f1_score": 0.94,  # 94% F1 score minimum
            "precision": 0.93,  # 93% precision minimum
            "recall": 0.93,  # 93% recall minimum
        }

    def test_model_size(self, trained_model):
        """Test if model size is within acceptable limits"""
        model_size_mb = os.path.getsize(trained_model) / (1024 * 1024)

        # Assert model is not too large (adjust threshold as needed)
        max_size_mb = 100  # Maximum allowed model size in MB
        assert (
            model_size_mb < max_size_mb
        ), f"Model size {model_size_mb:.2f}MB exceeds limit of {max_size_mb}MB"

        print(f"Model size: {model_size_mb:.2f}MB")

    def test_model_complexity(self, trained_model):
        """Test if model complexity is within acceptable limits"""
        model = tf.keras.models.load_model(trained_model)

        # Count number of parameters
        total_params = model.count_params()

        # Assert parameter count is within limits
        max_params = 20_000_000  # 20 million parameters
        assert (
            total_params < max_params
        ), f"Model has {total_params} parameters, exceeding limit of {max_params}"

        print(f"Model has {total_params:,} parameters")

    def test_inference_speed(self, trained_model):
        """Test model inference speed to ensure it meets performance requirements"""
        model = tf.keras.models.load_model(trained_model)

        # Create dummy batch for inference
        batch_size = 32
        dummy_input = np.random.random((batch_size, 95, 126, 3)).astype(np.float32)

        # Warm-up run
        _ = model.predict(dummy_input)

        # Measure inference time
        import time

        start_time = time.time()
        num_runs = 10

        for _ in range(num_runs):
            _ = model.predict(dummy_input)

        end_time = time.time()
        avg_inference_time = (end_time - start_time) / num_runs

        # Assert inference time is within acceptable limits
        max_inference_time = 0.5  # 500ms per batch
        assert (
            avg_inference_time < max_inference_time
        ), f"Average inference time {avg_inference_time*1000:.2f}ms exceeds limit of {max_inference_time*1000}ms"

        print(
            f"Average inference time: {avg_inference_time*1000:.2f}ms per batch of {batch_size}"
        )
        print(
            f"Average inference time per sample: {avg_inference_time*1000/batch_size:.2f}ms"
        )

    def test_prediction_consistency(self, trained_model):
        """Test if model predictions are consistent for the same input"""
        model = tf.keras.models.load_model(trained_model)

        # Create fixed dummy input
        np.random.seed(42)
        dummy_input = np.random.random((10, 95, 126, 3)).astype(np.float32)

        # Get predictions multiple times
        predictions_1 = model.predict(dummy_input)
        predictions_2 = model.predict(dummy_input)

        # Check if predictions are identical
        assert np.allclose(
            predictions_1, predictions_2, rtol=1e-5, atol=1e-7
        ), "Model predictions are not consistent"

    def test_numerical_stability(self, trained_model):
        """Test if model exhibits numerical stability with extreme inputs"""
        model = tf.keras.models.load_model(trained_model)

        # Test with very small values
        small_input = np.ones((5, 95, 126, 3)) * 1e-6
        small_predictions = model.predict(small_input)

        # Test with very large values
        large_input = np.ones((5, 95, 126, 3)) * 1e6
        large_predictions = model.predict(large_input)

        # Check that predictions don't contain NaN or infinity
        assert not np.any(
            np.isnan(small_predictions)
        ), "Model produces NaN with small inputs"
        assert not np.any(
            np.isnan(large_predictions)
        ), "Model produces NaN with large inputs"
        assert not np.any(
            np.isinf(small_predictions)
        ), "Model produces infinity with small inputs"
        assert not np.any(
            np.isinf(large_predictions)
        ), "Model produces infinity with large inputs"
