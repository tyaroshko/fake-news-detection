"""Configuration settings for fake news detection"""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ModelConfig:
    """Model configuration"""

    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_length: int = 512
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    """Training configuration"""

    output_dir: str = "./fake_news_model"
    logging_dir: str = "./logs"
    batch_size: int = 16
    num_epochs: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    gradient_accumulation_steps: int = 2
    fp16: bool = torch.cuda.is_available()
    seed: int = 42
    eval_strategy: str = (
        "steps"  # Updated from evaluation_strategy (new parameter name)
    )
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    early_stopping_patience: int = 3
    report_to: str = "wandb"  # Options: "none", "wandb", "tensorboard", "all"
    wandb_project: str = "fake-news-detection"
    use_gradient_checkpointing: bool = (
        False  # Enable to reduce memory usage at cost of speed
    )


@dataclass
class DataConfig:
    """Data configuration"""

    data_path: str = "WELFake_Dataset.csv"
    test_size: float = 0.15
    val_size: float = 0.15
    text_column: str = "text"
    label_column: str = "label"
    title_column: Optional[str] = "title"
    combine_title_text: bool = True
    random_state: int = 42
    sample_ratio: Optional[float] = (
        None  # If set, use only this fraction of data (e.g., 0.1 for 10%)
    )


@dataclass
class Config:
    """Main configuration"""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
