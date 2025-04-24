import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from src.models.classifier import VesselClassifier
from src.data.data_generator import VesselDataGenerator

from src.utils.evaluator import ModelEvaluator


class TestModelPipeline:
    """Integration tests for the full model training and evaluation pipeline"""

    @pytest.fixture
    def test_data_dir(self, tmp_path):
        """Create test data directory with minimal dataset for integration testing"""
        # Create test directory structure for train, validation, and test
        data_dir = tmp_path / "test_data"
        splits = ["train", "validation", "test"]
        methods = ["tfmel", "stft", "logscale"]
        classes = ["tug", "tanker", "cargo", "passengership", "background"]

        # Create directories and dummy spectrograms
        for split in splits:
            for method in methods:
                for class_name in classes:
                    class_dir = data_dir / split / method / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)

                    # Create 10 dummy spectrogram files per class
                    num_samples = 10 if split == "train" else 5
                    for i in range(num_samples):
                        # Create dummy spectrogram with shape (95, 126)
                        dummy_spec = np.random.random((95, 126)).astype(np.float32)
                        np.save(class_dir / f"sample_{i}.npy", dummy_spec)

        return data_dir

    @pytest.fixture
    def results_dir(self, tmp_path):
        """Create temporary results directory"""
        results = tmp_path / "results"
        results.mkdir()
        return results

    @pytest.fixture
    def model_params(self):
        """Test model parameters with reduced complexity for faster tests"""
        return {
            "initial_filters": 16,  # Smaller for faster tests
            "initial_kernel_size": 5,
            "initial_stride": 2,
            "block_filters": [16, 32],  # Reduced for testing
            "block_kernel_size": 3,
            "dropout_rate": 0.3,
            "l2_reg": 0.01,
            "use_augmentation": False,
        }

    @pytest.fixture
    def optimizer_params(self):
        """Test optimizer parameters"""
        return {"optimizer": "adam", "learning_rate": 0.001}

    @pytest.fixture
    def training_params(self, results_dir):
        """Test training parameters"""
        return {
            "epochs": 2,  # Just a couple epochs for testing
            "save_dir": str(results_dir),
            "early_stopping_patience": 5,
            "reduce_lr_patience": 2,
            "reduce_lr_factor": 0.5,
        }

    def test_full_pipeline(
        self,
        test_data_dir,
        results_dir,
        model_params,
        optimizer_params,
        training_params,
    ):
        """Test the complete model training and evaluation pipeline"""
        # Create data generators
        train_generator = VesselDataGenerator(
            data_path=test_data_dir / "train",
            preprocess_type=["tfmel", "stft", "logscale"],
            batch_size=8,
            shuffle=True,
        )

        val_generator = VesselDataGenerator(
            data_path=test_data_dir / "validation",
            preprocess_type=["tfmel", "stft", "logscale"],
            batch_size=8,
            shuffle=False,
        )

        test_generator = VesselDataGenerator(
            data_path=test_data_dir / "test",
            preprocess_type=["tfmel", "stft", "logscale"],
            batch_size=8,
            shuffle=False,
        )

        # Get input shape
        input_shape = train_generator.get_input_shape()

        # Create classifier
        classifier = VesselClassifier(
            input_shape=input_shape, num_classes=5, model_params=model_params
        )

        # Compile model
        classifier.compile_model(optimizer_params=optimizer_params)

        # Train model with minimal epochs
        history = classifier.train(
            train_generator, val_generator, training_params=training_params
        )

        # Verify training happened
        assert history is not None
        assert "accuracy" in history.history
        assert len(history.history["accuracy"]) > 0

        # Check if model file was saved
        model_path = results_dir / "best_model.keras"
        assert model_path.exists()

        # Test evaluation
        evaluator = ModelEvaluator(
            model=classifier.get_model(),
            test_generator=test_generator,
            history=history,
            save_dir=str(results_dir / "eval"),
        )

        metrics = evaluator.evaluate_model()

        # Check if metrics were calculated
        assert "f1_macro" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics

        # Create visualizations
        evaluator.plot_confusion_matrix()
        evaluator.plot_learning_curves()

        # Check if visualization files were created
        assert (results_dir / "eval" / "confusion_matrix.png").exists()
        assert (results_dir / "eval" / "learning_curves.png").exists()
