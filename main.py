import argparse
from pathlib import Path

import yaml

from src.data.data_generator import VesselDataGenerator
from src.models.classifier import VesselClassifier
from src.utils.evaluator import ModelEvaluator


def load_config(config_path):
    """Load YAML configuration file"""

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():

    parser = argparse.ArgumentParser(description="Train and evaluate vessel classifier")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    # If config file is provided, load it
    if args.config:
        config = load_config(args.config)

        # Extract parameters from config
        data_root = Path(config["data"]["root_path"])
        batch_size = config["data"]["batch_size"]
        preprocess_methods = config["data"]["preprocess_methods"]
        results_dir = Path(config["output"]["save_dir"])
        eval_dir = Path(config["output"]["eval_dir"])
        model_params = config["model"]["params"]
        optimizer_params = config["optimizer"]
        training_params = {
            "epochs": config["training"]["epochs"],
            "save_dir": str(results_dir),
            "early_stopping_patience": config["training"]["early_stopping_patience"],
            "reduce_lr_patience": config["training"]["reduce_lr_patience"],
            "reduce_lr_factor": config["training"]["reduce_lr_factor"],
            "monitor_metric": config["training"]["monitor_metric"],
            "monitor_mode": config["training"]["monitor_mode"],
        }
    else:
        # Default configuration if no config file is provided
        data_root = (
            Path(__file__).resolve().parents[0] / "datasets" / "VTUAD" / "inclusion_2000_exclusion_4000"
        )
        results_dir = Path("results") / Path("vessel_classification_results")
        eval_dir = Path("results") / Path("evaluation_results")
        preprocess_methods = ["tfmel", "stft", "logscale"]
        batch_size = 32

        # Default model parameters
        model_params = {
            "initial_filters": 64,
            "initial_kernel_size": 7,
            "initial_stride": 2,
            "block_filters": [64, 128, 256, 512],
            "block_kernel_size": 3,
            "dropout_rate": 0.3,
            "l2_reg": 0.01,
            "use_augmentation": True,
            "freq_mask_param": 10,
            "time_mask_param": 10,
        }

        # Default optimizer parameters
        optimizer_params = {
            "optimizer": "sgd",
            "learning_rate": 0.001,
            "momentum": 0.9,
            "nesterov": True,
            "clipnorm": 1.0,
        }

        # Default training parameters
        training_params = {
            "epochs": 200,
            "save_dir": str(results_dir),
            "early_stopping_patience": 30,
            "reduce_lr_patience": 20,
            "reduce_lr_factor": 0.5,
            "monitor_metric": "val_accuracy",
            "monitor_mode": "max",
        }

    # Create output directories
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using preprocessing methods: {preprocess_methods}")
    print(f"Data root path: {data_root}")

    # Create data generators
    try:
        train_generator = VesselDataGenerator(
            data_path=data_root / "train",
            preprocess_type=preprocess_methods,
            batch_size=batch_size,
            shuffle=True,
        )

        val_generator = VesselDataGenerator(
            data_path=data_root / "validation",
            preprocess_type=preprocess_methods,
            batch_size=batch_size,
            shuffle=False,
        )

        test_generator = VesselDataGenerator(
            data_path=data_root / "test",
            preprocess_type=preprocess_methods,
            batch_size=batch_size,
            shuffle=False,
        )
    except Exception as e:
        print(f"Error creating data generators: {str(e)}")
        return

    # Get input shape from generator
    input_shape = train_generator.get_input_shape()
    print(f"Input shape: {input_shape}")

    try:
        # Create classifier
        classifier = VesselClassifier(
            input_shape=input_shape, num_classes=5, model_params=model_params
        )

        # Compile model
        classifier.compile_model(optimizer_params=optimizer_params)

        # Train the model
        print("\nStarting model training...")
        history = classifier.train(
            train_generator, val_generator, training_params=training_params
        )

        # Evaluate the model
        print("\nEvaluating model...")
        evaluator = ModelEvaluator(
            model=classifier.get_model(),
            test_generator=test_generator,
            history=history,
            save_dir=str(eval_dir),
        )
        metrics = evaluator.create_full_report()

        print("\nTraining and evaluation completed successfully!")
        print(f"Results saved to {results_dir} and {eval_dir}")

    except Exception as e:
        print(f"Error during model training/evaluation: {str(e)}")
        return


if __name__ == "__main__":
    main()
