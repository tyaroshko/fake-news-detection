import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union

import litellm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from ..schemas.classification import LLMClassificationResult

logger = logging.getLogger(__name__)


class ClassificationService:
    def __init__(self):
        logger.info("ClassificationService initialized")
        self.models = {}
        self.tokenizers = {}
        self.available_models = {
            # Custom trained fake news detection model
            "roberta-base-fake-news": {
                "type": "bert",
                "description": "Custom RoBERTa model trained on WELFake dataset",
                "hf_path": "tyaroshko/roberta-base-fake-news-detection",
            },
            "distilbert-base-fake-news": {
                "type": "bert",
                "description": "Custom DistilBERT model trained on WELFake dataset",
                "hf_path": "tyaroshko/distilbert-base-uncased-fake-news-detection",
            },
            # # BERT-like models for fake news detection
            # "martin-ha/toxic-comment-model": {
            #     "type": "bert",
            #     "description": "BERT model fine-tuned for toxic content detection",
            # },
            # "cardiffnlp/twitter-roberta-base-sentiment": {
            #     "type": "bert",
            #     "description": "RoBERTa model for sentiment analysis",
            # },
            # "facebook/bart-large-mnli": {
            #     "type": "bert",
            #     "description": "BART model for natural language inference",
            # },
            # LLM models through litellm
            "gpt-5-mini": {
                "type": "llm",
                "description": "OpenAI GPT-5 Mini for text classification",
            },
            "claude-4-sonnet": {
                "type": "llm",
                "description": "Anthropic Claude 3.5 Sonnet for text classification",
            },
            "gemini-2.5-flash": {
                "type": "llm",
                "description": "Google Gemini 2.5 Flash for text classification",
            },
        }

    async def load_model(self, model_name: str) -> None:
        """Load a model if not already loaded."""
        if model_name in self.models:
            return

        model_config = self.available_models.get(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not available")

        if model_config["type"] == "bert":
            try:
                logger.info(f"Loading BERT model: {model_name}")
                # Check if it's a local model with a path specified
                model_path = model_config.get("hf_path", model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                logger.info(f"Successfully loaded model: {model_name}")
                logger.info(
                    f"Model label mapping: {model.config.id2label if hasattr(model.config, 'id2label') else 'No label mapping'}"
                )
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
        elif model_config["type"] == "llm":
            # LLM models are called via API, no local loading needed
            logger.info(f"LLM model {model_name} ready for API calls")

    async def classify_text_bert(self, text: str, model_name: str) -> Tuple[str, float]:
        """Classify text using BERT-like model."""
        if model_name not in self.models:
            await self.load_model(model_name)

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        # Tokenize and prepare input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

            prediction = "FAKE" if predicted_class == 0 else "REAL"

        return prediction, confidence

    async def classify_text_llm(
        self, text: str, model_name: str
    ) -> LLMClassificationResult:
        """Classify text using LLM through litellm."""
        prompt = f"""Analyze the following text and determine if it contains real news or fake news.
Respond with only "REAL" or "FAKE" followed by a confidence score between 0 and 1.

Text: {text[:2000]}  # Limit text length for API

Response format: REAL/FAKE,confidence_score"""

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                ),
            )
            logger.info(f"LLM response: {response}")
            result_text = response.choices[0].message.content.strip()
            parts = result_text.split(",")

            if len(parts) == 2:
                prediction = parts[0].strip().lower()
                try:
                    confidence = float(parts[1].strip())
                except ValueError:
                    confidence = 0.5
            else:
                prediction = "unknown"
                confidence = 0.5

            return LLMClassificationResult(result=prediction, confidence=confidence)

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return LLMClassificationResult(result="unknown", confidence=0.5)

    async def classify_text(
        self, text: str, model_name: str
    ) -> Union[Tuple[str, float], LLMClassificationResult]:
        """Main classification method that routes to appropriate model type."""
        model_config = self.available_models.get(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not available")

        if model_config["type"] == "bert":
            return await self.classify_text_bert(text, model_name)
        elif model_config["type"] == "llm":
            logger.info(f"Classifying text with LLM model: {model_name}")
            return await self.classify_text_llm(text, model_name)
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")

    def get_available_models(self) -> List[Dict]:
        """Get list of available models."""
        return [
            {"name": name, "type": config["type"], "description": config["description"]}
            for name, config in self.available_models.items()
        ]


# Global service instance
classification_service = ClassificationService()
