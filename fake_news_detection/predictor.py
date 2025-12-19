"""Prediction module"""

from typing import Any, Dict, List, Union

import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


class FakeNewsPredictor:
    """Class for making predictions"""

    def __init__(self, model_path: str):
        """
        Initialize predictor

        Args:
            model_path: Path to saved model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load model and tokenizer

        Args:
            model_path: Path to saved model
        """
        print(f"Loading model from {model_path}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully")

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict for a single text

        Args:
            text: Input text

        Returns:
            Prediction dictionary
        """
        # Tokenize
        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)

        # Get results
        label = self.model.config.id2label[predicted_class.item()]
        confidence = probabilities[0][predicted_class.item()].item()

        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "label": label,
            "confidence": confidence,
            "fake_probability": probabilities[0][0].item(),
            "real_probability": probabilities[0][1].item(),
        }

    def predict_batch(
        self, texts: List[str], batch_size: int = 32, return_dataframe: bool = True
    ) -> Union[List[Dict[str, Any]], pd.DataFrame]:
        """
        Predict for multiple texts

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            return_dataframe: Whether to return results as DataFrame

        Returns:
            List of predictions or DataFrame
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(probabilities, dim=-1)

            # Process results
            for j, text in enumerate(batch_texts):
                label = self.model.config.id2label[predicted_classes[j].item()]
                confidence = probabilities[j][predicted_classes[j].item()].item()

                results.append(
                    {
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "label": label,
                        "confidence": confidence,
                        "fake_probability": probabilities[j][0].item(),
                        "real_probability": probabilities[j][1].item(),
                    }
                )

        if return_dataframe:
            return pd.DataFrame(results)
        return results

    def predict_from_csv(
        self, csv_path: str, text_column: str = "text", output_path: str = None
    ) -> pd.DataFrame:
        """
        Predict from CSV file

        Args:
            csv_path: Path to CSV file
            text_column: Name of text column
            output_path: Path to save predictions (optional)

        Returns:
            DataFrame with predictions
        """
        # Load data
        df = pd.read_csv(csv_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")

        # Get predictions
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts, return_dataframe=False)

        # Add predictions to dataframe
        for key in ["label", "confidence", "fake_probability", "real_probability"]:
            df[f"predicted_{key}"] = [p[key] for p in predictions]

        # Save if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")

        return df
