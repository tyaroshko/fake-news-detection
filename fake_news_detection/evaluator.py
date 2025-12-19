"""Model evaluation module"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)


class FakeNewsEvaluator:
    """Class for evaluating fake news detection model"""

    def __init__(self):
        """Initialize evaluator"""
        self.results = {}

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model predictions

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Prediction probabilities (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = (
            precision_recall_fscore_support(y_true, y_pred, average=None)
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Classification report
        report = classification_report(
            y_true, y_pred, target_names=["Fake", "Real"], output_dict=True
        )

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "classification_report": report,
            "per_class": {
                "fake": {
                    "precision": precision_per_class[0],
                    "recall": recall_per_class[0],
                    "f1": f1_per_class[0],
                    "support": int(support_per_class[0]),
                },
                "real": {
                    "precision": precision_per_class[1],
                    "recall": recall_per_class[1],
                    "f1": f1_per_class[1],
                    "support": int(support_per_class[1]),
                },
            },
        }

        # ROC and PR curves if probabilities are provided
        if y_probs is not None:
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            roc_auc = auc(fpr, tpr)

            # Precision-Recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_true, y_probs[:, 1]
            )

            results["roc"] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

            results["pr_curve"] = {"precision": precision_curve, "recall": recall_curve}

        self.results = results
        return results

    def get_predictions_from_trainer(self, predictions):
        """
        Extract predictions from trainer output

        Args:
            predictions: Trainer prediction output

        Returns:
            Tuple of (y_pred, y_true, y_probs)
        """
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        y_probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

        return y_pred, y_true, y_probs

    def print_summary(self, results: Optional[Dict[str, Any]] = None):
        """
        Print evaluation summary

        Args:
            results: Evaluation results (uses stored results if None)
        """
        if results is None:
            results = self.results

        if not results:
            print("No results to display")
            return

        print("" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        # Overall metrics
        print("Overall Metrics:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1']:.4f}")

        if "roc" in results:
            print(f"  ROC AUC:   {results['roc']['auc']:.4f}")

        # Per-class metrics
        print("Per-Class Metrics:")
        for class_name in ["fake", "real"]:
            metrics = results["per_class"][class_name]
            print(f"{class_name.capitalize()}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1']:.4f}")
            print(f"    Support:   {metrics['support']}")

        # Confusion Matrix
        print("Confusion Matrix:")
        cm = results["confusion_matrix"]
        print("          Predicted")
        print("          Fake  Real")
        print(f"Actual Fake  {cm[0, 0]:5d} {cm[0, 1]:5d}")
        print(f"       Real  {cm[1, 0]:5d} {cm[1, 1]:5d}")

        print("" + "=" * 60)

    def save_results(self, filepath: str):
        """Save evaluation results"""
        import json

        # Convert numpy arrays to lists for JSON serialization
        save_results = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                save_results[key] = value.tolist()
            elif key in ["roc", "pr_curve"]:
                save_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                save_results[key] = value

        with open(filepath, "w") as f:
            json.dump(save_results, f, indent=4)
        print(f"Results saved to {filepath}")
