from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ModelEvaluator:
    """Class for evaluating and creating reports for the vessel classifier"""

    def __init__(self, model, test_generator, history=None, save_dir="./results"):
        """Initialize the evaluator

        Args:
            model: Trained TensorFlow model
            test_generator: Data generator for test set
            history: Training history from model.fit
            save_dir: Directory to save evaluation results
        """
        self.model = model
        self.test_generator = test_generator
        self.history = history
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = list(test_generator.class_mapping.keys())

    def evaluate_model(self):
        """Evaluate model on test set and return metrics"""
        # Get predictions
        y_pred = []
        y_true = []

        for i in range(len(self.test_generator)):
            x, y = self.test_generator[i]
            pred = self.model.predict(x)
            y_pred.extend(np.argmax(pred, axis=1))
            y_true.extend(np.argmax(y, axis=1))

        # Calculate metrics
        self.confusion_mat = confusion_matrix(y_true, y_pred)
        self.classification_rep = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )

        # Calculate additional metrics
        metrics = {
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
        }

        return metrics

    def plot_learning_curves(self):
        """Plot training and validation learning curves"""
        if self.history is None:
            raise ValueError("No training history provided")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        ax1.plot(self.history.history["accuracy"], label="Training")
        ax1.plot(self.history.history["val_accuracy"], label="Validation")
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True)

        # Plot loss
        ax2.plot(self.history.history["loss"], label="Training")
        ax2.plot(self.history.history["val_loss"], label="Validation")
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(self.save_dir / "learning_curves.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(self):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))

        # Normalize confusion matrix
        conf_mat_norm = (
            self.confusion_mat.astype("float")
            / self.confusion_mat.sum(axis=1)[:, np.newaxis]
        )

        sns.heatmap(
            conf_mat_norm,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )

        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(
            self.save_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def save_classification_report(self):
        """Save classification report as CSV"""
        df = pd.DataFrame(self.classification_rep).transpose()
        df.to_csv(self.save_dir / "classification_report.csv")

    def create_full_report(self):
        """Create and save complete evaluation report"""
        print("Evaluating model on test set...")
        metrics = self.evaluate_model()

        print("\nGenerating visualizations and reports...")
        self.plot_confusion_matrix()
        if self.history is not None:
            self.plot_learning_curves()

        # Calculate test accuracy using predictions
        total_correct = 0
        total_samples = 0

        for i in range(len(self.test_generator)):
            x, y = self.test_generator[i]
            pred = self.model.predict(x)
            pred_classes = np.argmax(pred, axis=1)
            true_classes = np.argmax(y, axis=1)
            total_correct += np.sum(pred_classes == true_classes)
            total_samples += len(true_classes)

        test_acc = total_correct / total_samples
        print(f"\nTest Accuracy: {test_acc:.4f}")

        # Add test accuracy to metrics
        metrics["test_accuracy"] = test_acc

        # Create a metrics summary file
        metrics_summary = [
            "Test Set Metrics Summary",
            "====================",
            f"Test Accuracy: {test_acc:.4f}",
            f"Macro F1-Score: {metrics['f1_macro']:.4f}",
            f"Macro Precision: {metrics['precision_macro']:.4f}",
            f"Macro Recall: {metrics['recall_macro']:.4f}",
        ]

        # Save metrics summary
        with open(self.save_dir / "test_metrics_summary.txt", "w") as f:
            f.write("\n".join(metrics_summary))

        # Save detailed classification report
        self.save_classification_report()

        return metrics
