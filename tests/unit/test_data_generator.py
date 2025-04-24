import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from src.data.data_generator import VesselDataGenerator


class TestVesselDataGenerator:
    """Test suite for the VesselDataGenerator class"""

    @pytest.fixture
    def test_data_dir(self, tmp_path):
        """Fixture to create a temporary test data directory with dummy spectrograms"""
        # Create test directory structure
        data_dir = tmp_path / "test_data"
        methods = ["tfmel", "stft", "logscale"]
        classes = ["tug", "tanker", "cargo", "passengership", "background"]

        # Create directories and dummy spectrograms
        for method in methods:
            for class_name in classes:
                class_dir = data_dir / method / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                # Create 5 dummy spectrogram files per class
                for i in range(5):
                    # Create dummy spectrogram with shape (95, 126)
                    dummy_spec = np.random.random((95, 126)).astype(np.float32)
                    np.save(class_dir / f"sample_{i}.npy", dummy_spec)

        return data_dir

    def test_single_method_generator(self, test_data_dir):
        """Test data generator with a single preprocessing method"""
        batch_size = 4
        generator = VesselDataGenerator(
            data_path=test_data_dir,
            preprocess_type="tfmel",
            batch_size=batch_size,
            shuffle=True,
        )

        # Check basic properties
        assert len(generator) == 6  # 25 samples with batch size 4 -> 6.25 -> 6 batches

        # Get a batch
        X, y = generator[0]

        # Check shapes
        assert X.shape == (batch_size, 95, 126, 1)
        assert y.shape == (batch_size, 5)

        # Check one-hot encoding
        assert np.all(np.sum(y, axis=1) == 1)

    def test_multiple_methods_generator(self, test_data_dir):
        """Test data generator with multiple preprocessing methods"""
        batch_size = 4
        generator = VesselDataGenerator(
            data_path=test_data_dir,
            preprocess_type=["tfmel", "stft", "logscale"],
            batch_size=batch_size,
            shuffle=True,
        )

        # Get a batch
        X, y = generator[0]

        # Check shapes with 3 channels
        assert X.shape == (batch_size, 95, 126, 3)
        assert y.shape == (batch_size, 5)

    def test_get_input_shape(self, test_data_dir):
        """Test get_input_shape method"""
        # Single method
        single_generator = VesselDataGenerator(
            data_path=test_data_dir, preprocess_type="tfmel", batch_size=4
        )
        assert single_generator.get_input_shape() == (95, 126, 1)

        # Multiple methods
        multi_generator = VesselDataGenerator(
            data_path=test_data_dir, preprocess_type=["tfmel", "stft"], batch_size=4
        )
        assert multi_generator.get_input_shape() == (95, 126, 2)
