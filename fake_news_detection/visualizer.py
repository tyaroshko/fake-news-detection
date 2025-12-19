"""Visualization module"""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class FakeNewsVisualizer:
    """Class for creating visualizations"""

    def __init__(self, style: str = "ggplot"):
        """
        Initialize visualizer

        Args:
            style: Matplotlib style (default, ggplot, seaborn-v0_8, etc.)
        """
        # Set matplotlib style
        if style != "default":
            plt.style.use(style)

        # Set seaborn style and palette
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    def plot_training_history(self, trainer) -> None:
        """
        Plot training history from trainer

        Args:
            trainer: HuggingFace Trainer object
        """
        history = trainer.state.log_history

        train_loss = []
        eval_loss = []
        eval_accuracy = []
        eval_f1 = []
        epochs = []

        for entry in history:
            if "loss" in entry and "epoch" in entry:
                train_loss.append(entry["loss"])
            if "eval_loss" in entry:
                eval_loss.append(entry["eval_loss"])
                epochs.append(entry["epoch"])
                if "eval_accuracy" in entry:
                    eval_accuracy.append(entry["eval_accuracy"])
                if "eval_f1" in entry:
                    eval_f1.append(entry["eval_f1"])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Training loss
        if train_loss:
            axes[0, 0].plot(
                train_loss, label="Training Loss", color="#2E86AB", alpha=0.7
            )
            axes[0, 0].set_xlabel("Steps")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].set_title("Training Loss over Steps")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Validation loss
        if eval_loss and epochs:
            axes[0, 1].plot(
                epochs, eval_loss, label="Validation Loss", marker="o", color="#A23B72"
            )
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].set_title("Validation Loss over Epochs")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Validation accuracy
        if eval_accuracy and epochs:
            axes[1, 0].plot(
                epochs,
                eval_accuracy,
                label="Validation Accuracy",
                marker="s",
                color="#F18F01",
            )
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Accuracy")
            axes[1, 0].set_title("Validation Accuracy over Epochs")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Validation F1
        if eval_f1 and epochs:
            axes[1, 1].plot(
                epochs, eval_f1, label="Validation F1", marker="^", color="#C73E1D"
            )
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("F1 Score")
            axes[1, 1].set_title("Validation F1 Score over Epochs")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=100, bbox_inches="tight")
        plt.show()

    def plot_confusion_matrix(
        self, cm: np.ndarray, labels: List[str] = ["Fake", "Real"]
    ) -> None:
        """
        Plot confusion matrix

        Args:
            cm: Confusion matrix
            labels: Class labels
        """
        plt.figure(figsize=(8, 6))

        # Normalize confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Create annotation text
        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f"{cm[i, j]}({cm_normalized[i, j]:.2%})"

        sns.heatmap(
            cm,
            annot=annotations,
            fmt="",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
        plt.show()

    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> None:
        """
        Plot ROC curve

        Args:
            fpr: False positive rate
            tpr: True positive rate
            roc_auc: Area under ROC curve
        """
        plt.figure(figsize=(8, 6))

        plt.plot(
            fpr, tpr, color="#2E86AB", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        plt.plot([0, 1], [0, 1], color="#A23B72", lw=2, linestyle="--", alpha=0.7)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.savefig("roc_curve.png", dpi=100, bbox_inches="tight")
        plt.show()

    def plot_precision_recall_curve(
        self, precision: np.ndarray, recall: np.ndarray
    ) -> None:
        """
        Plot precision-recall curve

        Args:
            precision: Precision values
            recall: Recall values
        """
        plt.figure(figsize=(8, 6))

        plt.plot(recall, precision, color="#F18F01", lw=2)
        plt.fill_between(recall, precision, alpha=0.2, color="#F18F01")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.savefig("precision_recall_curve.png", dpi=100, bbox_inches="tight")
        plt.show()

    def plot_classification_report_heatmap(self, report: Dict[str, Any]) -> None:
        """
        Plot classification report as heatmap

        Args:
            report: Classification report dictionary
        """
        # Convert to DataFrame
        df_report = pd.DataFrame(report).transpose()
        df_report = df_report.iloc[:-3, :-1]  # Remove support column and summary rows

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            df_report, annot=True, cmap="YlGnBu", fmt=".3f", cbar_kws={"label": "Score"}
        )
        plt.title("Classification Report Heatmap")
        plt.savefig("classification_report.png", dpi=100, bbox_inches="tight")
        plt.show()

    def plot_label_distribution(self, datasets: Dict[str, Any]) -> None:
        """
        Plot label distribution across datasets

        Args:
            datasets: Dictionary of datasets
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ["#A23B72", "#F18F01"]

        for idx, (split_name, dataset) in enumerate(datasets.items()):
            if split_name in ["train", "validation", "test"]:
                labels = dataset["label"]
                unique, counts = np.unique(labels, return_counts=True)

                bars = axes[idx].bar(["Fake", "Real"], counts, color=colors, alpha=0.7)
                axes[idx].set_title(f"{split_name.capitalize()} Set Distribution")
                axes[idx].set_ylabel("Count")
                axes[idx].set_xlabel("Label")

                # Add percentage labels
                total = sum(counts)
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    axes[idx].text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + total * 0.01,
                        f"{count}({count / total * 100:.1f}%)",
                        ha="center",
                        va="bottom",
                    )

        plt.tight_layout()
        plt.savefig("label_distribution.png", dpi=100, bbox_inches="tight")
        plt.show()

    def plot_all_metrics(self, results: Dict[str, Any]) -> None:
        """
        Create all visualization plots

        Args:
            results: Evaluation results dictionary
        """
        # Confusion Matrix
        if "confusion_matrix" in results:
            self.plot_confusion_matrix(results["confusion_matrix"])

        # ROC Curve
        if "roc" in results:
            self.plot_roc_curve(
                results["roc"]["fpr"], results["roc"]["tpr"], results["roc"]["auc"]
            )

        # Precision-Recall Curve
        if "pr_curve" in results:
            self.plot_precision_recall_curve(
                results["pr_curve"]["precision"], results["pr_curve"]["recall"]
            )

        # Classification Report
        if "classification_report" in results:
            self.plot_classification_report_heatmap(results["classification_report"])
