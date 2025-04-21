from pathlib import Path

import keras

from src.models.spec_augmentation import SpectrogramAugmentation


class VesselClassifier:
    def __init__(self, input_shape, num_classes=5, model_params=None):
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Default model parameters
        self.model_params = {
            "initial_filters": 64,
            "initial_kernel_size": 7,
            "initial_stride": 2,
            "block_filters": [64, 128, 256, 512],
            "block_kernel_size": 3,
            "dropout_rate": 0.3,
            "l2_reg": 0.01,
            "use_augmentation": False,
            "freq_mask_param": 10,
            "time_mask_param": 10,
        }

        if model_params is not None:
            self.model_params.update(model_params)

        self.model = self._create_resnet18()

    def _resnet_block(self, x, filters, kernel_size=3, stride=1, conv_shortcut=True):
        regularizer = keras.regularizers.l2(self.model_params["l2_reg"])
        shortcut = x

        if conv_shortcut:
            shortcut = keras.layers.Conv2D(
                filters,
                1,
                strides=stride,
                padding="same",
                kernel_regularizer=regularizer,
            )(shortcut)
            shortcut = keras.layers.BatchNormalization()(shortcut)

        x = keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=stride,
            padding="same",
            kernel_regularizer=regularizer,
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(self.model_params["dropout_rate"])(x)

        x = keras.layers.Conv2D(
            filters, kernel_size, padding="same", kernel_regularizer=regularizer
        )(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Add()([shortcut, x])
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(self.model_params["dropout_rate"])(x)

        return x

    def _create_resnet18(self):
        regularizer = keras.regularizers.l2(self.model_params["l2_reg"])
        inputs = keras.Input(shape=self.input_shape)

        # Add augmentation if enabled
        if self.model_params["use_augmentation"]:
            x = SpectrogramAugmentation(
                freq_mask_param=self.model_params["freq_mask_param"],
                time_mask_param=self.model_params["time_mask_param"],
            )(inputs)
        else:
            x = inputs

        # Initial convolution
        x = keras.layers.Conv2D(
            self.model_params["initial_filters"],
            self.model_params["initial_kernel_size"],
            strides=self.model_params["initial_stride"],
            padding="same",
            kernel_regularizer=regularizer,
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # ResNet blocks
        for filters in self.model_params["block_filters"]:
            x = self._resnet_block(
                x,
                filters,
                kernel_size=self.model_params["block_kernel_size"],
                stride=2 if filters > 64 else 1,
            )
            x = self._resnet_block(
                x,
                filters,
                kernel_size=self.model_params["block_kernel_size"],
                conv_shortcut=False,
            )

        # Final layers
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(self.model_params["dropout_rate"])(x)
        outputs = keras.layers.Dense(
            self.num_classes, activation="softmax", kernel_regularizer=regularizer
        )(x)

        return keras.Model(inputs, outputs)

    def train(self, train_generator, validation_generator, training_params=None):
        """Train the model

        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            training_params: Dictionary of training parameters

        Returns:
            Training history
        """
        # Default training parameters
        default_params = {
            "epochs": 50,
            "save_dir": "./results",
            "early_stopping_patience": 10,
            "reduce_lr_patience": 5,
            "reduce_lr_factor": 0.8,
            "save_best_only": True,
            "monitor_metric": "val_accuracy",
            "monitor_mode": "max",
        }

        # Update with user provided parameters if any
        if training_params is not None:
            default_params.update(training_params)

        save_dir = Path(default_params["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create callbacks
        callbacks = [
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                save_dir / "best_model.keras",
                monitor=default_params["monitor_metric"],
                mode=default_params["monitor_mode"],
                save_best_only=default_params["save_best_only"],
                verbose=1,
            ),
            # Learning rate scheduler
            keras.callbacks.ReduceLROnPlateau(
                monitor=default_params["monitor_metric"],
                factor=default_params["reduce_lr_factor"],
                patience=default_params["reduce_lr_patience"],
                verbose=1,
                min_delta=1e-4,
                min_lr=1e-6,
            ),
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor=default_params["monitor_metric"],
                mode=default_params["monitor_mode"],
                min_delta=1e-4,
                patience=default_params["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            # CSV Logger for training history
            keras.callbacks.CSVLogger(
                save_dir / "training_history.csv", separator=",", append=False
            ),
        ]

        # Train model
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=default_params["epochs"],
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def compile_model(self, optimizer_params=None):
        """Compile the model with optimizer and loss function

        Args:
            optimizer_params: Dictionary of optimizer parameters
        """
        # Default optimizer parameters
        default_params = {
            "optimizer": "sgd",
            "learning_rate": 0.001,
            "momentum": 0.9,
            "nesterov": True,  # Enable Nesterov momentum
            "clipnorm": 1.0,  # Gradient clipping
        }

        # Update with user provided parameters if any
        if optimizer_params is not None:
            default_params.update(optimizer_params)

        # Create optimizer
        if default_params["optimizer"].lower() == "sgd":
            optimizer = keras.optimizers.SGD(
                learning_rate=default_params["learning_rate"],
                momentum=default_params["momentum"],
                nesterov=default_params["nesterov"],
                clipnorm=default_params["clipnorm"],
            )
        elif default_params["optimizer"].lower() == "adam":
            optimizer = keras.optimizers.Adam(
                learning_rate=default_params["learning_rate"],
                clipnorm=default_params["clipnorm"],
            )
        else:
            raise ValueError(f"Unsupported optimizer: {default_params['optimizer']}")

        # Additional metrics beyond accuracy
        metrics = [
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ]

        self.model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics
        )

    def predict(self, x):
        """Make predictions on input data

        Args:
            x: Input data to predict on

        Returns:
            Model predictions
        """
        return self.model.predict(x)

    def evaluate(self, x, y):
        """Evaluate model on test data

        Args:
            x: Test data
            y: True labels

        Returns:
            Evaluation metrics
        """
        return self.model.evaluate(x, y)

    def load_weights(self, weights_path):
        """Load model weights from file

        Args:
            weights_path: Path to weights file
        """
        self.model.load_weights(weights_path)

    def save_weights(self, weights_path):
        """Save model weights to file

        Args:
            weights_path: Path to save weights
        """
        self.model.save_weights(weights_path)

    def save_model(self, model_path):
        """Save complete model to file

        Args:
            model_path: Path to save model
        """
        self.model.save(model_path)

    def get_model(self):
        """Return the underlying Keras model

        Returns:
            The Keras model
        """
        return self.model
