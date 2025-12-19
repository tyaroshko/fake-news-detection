"""Model training module"""

from typing import Any, Dict, Optional

import evaluate
import numpy as np
import torch
from config import ModelConfig, TrainingConfig
from huggingface_hub import HfApi, whoami
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from utils import ensure_dir


class FakeNewsModel:
    """Class for fake news detection model"""

    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        """
        Initialize model

        Args:
            model_config: Model configuration
            training_config: Training configuration
        """
        self.model_config = model_config
        self.training_config = training_config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def initialize_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        print(f"Initializing model: {self.model_config.model_name}")

        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.model_config.model_name
        )

        # Initialize model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_config.model_name,
            num_labels=self.model_config.num_labels,
            id2label={0: "FAKE", 1: "REAL"},
            label2id={"FAKE": 0, "REAL": 1},
        )

        # Set dropout if specified
        if hasattr(self.model.config, "dropout"):
            self.model.config.dropout = self.model_config.dropout
        if hasattr(self.model.config, "attention_dropout"):
            self.model.config.attention_dropout = self.model_config.dropout

        # Enable gradient checkpointing if configured (reduces memory usage)
        if self.training_config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")

        print(f"Model parameters: {self.model.num_parameters():,}")

    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation

        Args:
            eval_pred: EvalPrediction object

        Returns:
            Dictionary of metrics
        """
        # Load metrics
        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        precision = precision_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
        recall = recall_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
        f1 = f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )

        return {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1": f1["f1"],
        }

    def create_training_args(self) -> TrainingArguments:
        """
        Create training arguments

        Returns:
            TrainingArguments object
        """
        ensure_dir(self.training_config.output_dir)
        ensure_dir(self.training_config.logging_dir)

        return TrainingArguments(
            output_dir=self.training_config.output_dir,
            learning_rate=self.training_config.learning_rate,
            per_device_train_batch_size=self.training_config.batch_size,
            per_device_eval_batch_size=self.training_config.batch_size,
            num_train_epochs=self.training_config.num_epochs,
            weight_decay=self.training_config.weight_decay,
            eval_strategy=self.training_config.eval_strategy,  # Updated from evaluation_strategy
            eval_steps=self.training_config.eval_steps,
            save_strategy=self.training_config.save_strategy,
            save_steps=self.training_config.save_steps,
            logging_dir=self.training_config.logging_dir,
            logging_steps=self.training_config.logging_steps,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            push_to_hub=True,
            report_to=self.training_config.report_to,  # Now configurable (none/wandb/tensorboard)
            wandb_project=self.training_config.wandb_project,
            warmup_steps=self.training_config.warmup_steps,
            fp16=self.training_config.fp16,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            seed=self.training_config.seed,
            save_total_limit=2,
            # Note: gradient_checkpointing removed - use model.gradient_checkpointing_enable() if needed
        )

    def train(self, train_dataset, val_dataset):
        """
        Train the model

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset

        Returns:
            Training results
        """
        if self.model is None or self.tokenizer is None:
            self.initialize_model_and_tokenizer()

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Training arguments
        training_args = self.create_training_args()

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience
                )
            ],
        )

        print("Starting training...")
        train_result = self.trainer.train()

        # Save the model
        self.trainer.save_model()
        self.trainer.save_state()

        print(f"Model saved to {self.training_config.output_dir}")

        return train_result

    def evaluate(self, dataset) -> Dict[str, float]:
        """
        Evaluate the model

        Args:
            dataset: Dataset to evaluate on

        Returns:
            Dictionary of metrics
        """
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")

        return self.trainer.evaluate(dataset)

    def predict(self, dataset):
        """
        Make predictions on a dataset

        Args:
            dataset: Dataset to predict on

        Returns:
            Predictions object
        """
        if self.trainer is None:
            raise ValueError("Model must be trained before prediction")

        return self.trainer.predict(dataset)

    def save_model(self, path: str):
        """Save model and tokenizer"""
        if self.model is not None:
            self.model.save_pretrained(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model and tokenizer"""
        self.tokenizer = DistilBertTokenizer.from_pretrained(path)
        self.model = DistilBertForSequenceClassification.from_pretrained(path)
        print(f"Model loaded from {path}")

    def push_to_hub(self, repo_name: str):
        """
        Push model to Hugging Face Hub

        Args:
            repo_name: Name of the repository (username/repo-name or repo-name)

        Raises:
            ValueError: If not logged in or authentication fails
        """
        # Check if logged in to Hugging Face
        try:
            user_info = whoami()
            username = user_info["name"]
            print(f"✓ Logged in as: {username}")
        except Exception as e:
            raise ValueError(
                f"Not logged in to Hugging Face. Please run 'huggingface-cli login' first. Error: {e}"
            )

        # Initialize HF API
        api = HfApi()

        # Parse repo_name to get full repository ID
        if "/" not in repo_name:
            full_repo_name = f"{username}/{repo_name}"
        else:
            full_repo_name = repo_name

        # Check if repository already exists
        try:
            repo_info = api.repo_info(repo_id=full_repo_name, repo_type="model")
            print(f"⚠ Repository '{full_repo_name}' already exists.")
            print(f"  URL: https://huggingface.co/{full_repo_name}")
            response = input("Do you want to overwrite it? (yes/no): ").strip().lower()
            if response not in ["yes", "y"]:
                print("Push cancelled.")
                return
        except RepositoryNotFoundError:
            print(
                f"✓ Repository '{full_repo_name}' does not exist. Creating new repository..."
            )

        # Push model and tokenizer
        print(f"Pushing model to {full_repo_name}...")
        self.model.push_to_hub(full_repo_name)
        self.tokenizer.push_to_hub(full_repo_name)
        print(
            f"✓ Model successfully pushed to Hugging Face Hub: https://huggingface.co/{full_repo_name}"
        )
