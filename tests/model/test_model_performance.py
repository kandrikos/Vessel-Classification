import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from src.models.classifier import VesselClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


class TestModelPerformance:
    """Tests for model performance benchmarking"""

    @pytest.fixture
    def test_dataset(self, tmp_path):
        """Create or load test dataset"""
        # In a real-world scenario, this would load your actual test data
        # For this example, we'll create synthetic data
        X = np.random.random((100, 95, 126, 3)).astype(np.float32)

        # Create synthetic labels (5 classes)
        y_true = np.random.randint(0, 5, 100)
        y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=5)

        return X, y_true, y_true_onehot

    @pytest.fixture
    def performance_thresholds(self):
        """Define performance thresholds for the model"""
        return {"accuracy": 0.95, "precision": 0.94, "recall": 0.94, "f1": 0.94}

    @pytest.fixture
    def performance_history(self, tmp_path):
        """Create or load performance history file"""
        history_file = tmp_path / "performance_history.json"

        # Create example history if it doesn't exist
        if not history_file.exists():
            history = {
                "commits": [
                    {
                        "commit_id": "abc123",
                        "timestamp": "2023-01-01",
                        "metrics": {
                            "accuracy": 0.96,
                            "precision": 0.95,
                            "recall": 0.95,
                            "f1": 0.95,
                        },
                    }
                ]
            }
            with open(history_file, "w") as f:
                json.dump(history, f)

        return history_file

    def test_performance_against_thresholds(
        self, test_dataset, performance_thresholds, trained_model
    ):
        """Test if model performance meets minimum thresholds"""
        X_test, y_true, _ = test_dataset

        # Load model
        model = tf.keras.models.load_model(trained_model)

        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")

        # Print metrics for report
        print(f"Model Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Compare against thresholds
        assert (
            accuracy >= performance_thresholds["accuracy"]
        ), f"Accuracy {accuracy:.4f} below threshold {performance_thresholds['accuracy']}"

        assert (
            precision >= performance_thresholds["precision"]
        ), f"Precision {precision:.4f} below threshold {performance_thresholds['precision']}"

        assert (
            recall >= performance_thresholds["recall"]
        ), f"Recall {recall:.4f} below threshold {performance_thresholds['recall']}"

        assert (
            f1 >= performance_thresholds["f1"]
        ), f"F1 score {f1:.4f} below threshold {performance_thresholds['f1']}"

    def test_no_performance_regression(
        self, test_dataset, performance_history, trained_model
    ):
        """Test that model performance hasn't regressed compared to history"""
        X_test, y_true, _ = test_dataset

        # Load model
        model = tf.keras.models.load_model(trained_model)

        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        current_metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro"),
            "recall": recall_score(y_true, y_pred, average="macro"),
            "f1": f1_score(y_true, y_pred, average="macro"),
        }

        # Load performance history
        with open(performance_history, "r") as f:
            history = json.load(f)

        # Get latest historical metrics
        latest_metrics = history["commits"][-1]["metrics"]

        # Allow a small regression tolerance (0.5%)
        tolerance = 0.005

        # Compare with historical performance
        for metric_name, current_value in current_metrics.items():
            historical_value = latest_metrics[metric_name]
            assert current_value >= (
                historical_value - tolerance
            ), f"{metric_name} regressed from {historical_value:.4f} to {current_value:.4f}"

        # Add current metrics to history (in a real scenario, this would be done in the CI pipeline)
        # For this test, we'll just print the metrics
        print(f"Current metrics have been validated against historical data")
