from pathlib import Path

import numpy as np
import tensorflow as tf


class VesselDataGenerator(tf.keras.utils.Sequence):
    """Data generator for vessel classification"""

    def __init__(self, data_path, preprocess_type, batch_size=32, shuffle=True):
        """Initialize the data generator

        Args:
            data_path: Path to preprocessed data directory
            preprocess_type: Either a single method ('cqt', 'mel', 'gammatone', 'tfmel', 'stft', 'logscale')
                           or a list of methods to combine
            batch_size: Size of batches to generate
            shuffle: Whether to shuffle data after each epoch
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_mapping = {
            "tug": 0,
            "tanker": 1,
            "cargo": 2,
            "passengership": 3,
            "background": 4,
        }

        # List of all available preprocessing methods
        self.available_methods = [
            "cqt",
            "mel",
            "gammatone",
            "tfmel",
            "stft",
            "logscale",
        ]

        # Handle preprocess_type input
        if isinstance(preprocess_type, list):
            # Multiple methods case
            for method in preprocess_type:
                if method not in self.available_methods:
                    raise ValueError(
                        f"Invalid method '{method}'. Available methods are: {self.available_methods}"
                    )
            self.methods_to_use = preprocess_type
            self.multiple_methods = True
        else:
            # Single method case
            if preprocess_type not in self.available_methods:
                raise ValueError(
                    f"Invalid preprocessing type. Available types are: {self.available_methods}"
                )
            self.methods_to_use = [preprocess_type]
            self.multiple_methods = False

        # Get list of all files
        if self.multiple_methods:
            # Use first method's directory as reference
            self.files = []
            for class_name in self.class_mapping.keys():
                class_path = self.data_path / self.methods_to_use[0] / class_name
                if class_path.exists():
                    self.files.extend(list(class_path.glob("*.npy")))

            # Create parallel paths for all methods
            self.all_method_files = {}
            for method in self.methods_to_use:
                self.all_method_files[method] = [
                    p.parent.parent.parent / method / p.parent.name / p.name
                    for p in self.files
                ]
        else:
            self.files = []
            for class_name in self.class_mapping.keys():
                class_path = self.data_path / self.methods_to_use[0] / class_name
                if class_path.exists():
                    self.files.extend(list(class_path.glob("*.npy")))

        self.indexes = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Return the number of batches per epoch"""
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Generate data
        X = []
        y = []

        for idx in indexes:
            if self.multiple_methods:
                # Load all specified preprocessing types
                method_data = []
                for method in self.methods_to_use:
                    data = np.load(self.all_method_files[method][idx])
                    method_data.append(data)

                # Stack all spectrograms along the channel dimension
                data = np.stack(method_data, axis=-1)
            else:
                # Load single preprocessing type
                data = np.load(self.files[idx])
                data = data[..., np.newaxis]  # Add channel dimension

            X.append(data)

            # Get class from parent directory name
            class_name = self.files[idx].parent.name
            y.append(self.class_mapping[class_name])

        X = np.array(X)
        y = tf.keras.utils.to_categorical(y, num_classes=len(self.class_mapping))

        return X, y

    def on_epoch_end(self):
        """Called at the end of every epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_input_shape(self):
        """Return the shape of input data"""
        return (95, 126, len(self.methods_to_use))
