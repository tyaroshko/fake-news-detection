"""Data loading and preprocessing module"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from config import DataConfig
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer


class FakeNewsDataLoader:
    """Class for loading and preprocessing fake news data"""

    def __init__(self, config: DataConfig):
        """
        Initialize data loader

        Args:
            config: Data configuration
        """
        self.config = config
        self.tokenizer = None
        self.df = None
        self.datasets = None

    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the WELFake dataset

        Returns:
            Preprocessed DataFrame
        """
        print(f"Loading data from {self.config.data_path}...")

        # Load CSV file
        df = pd.read_csv(self.config.data_path)

        # Display initial info
        print(f"Initial dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Handle column names (case-insensitive)
        df.columns = df.columns.str.lower()

        # Remove serial number column if exists
        if "serial" in df.columns or "unnamed: 0" in df.columns:
            df = df.drop(
                columns=["serial"] if "serial" in df.columns else ["unnamed: 0"]
            )

        # Check for required columns
        if self.config.text_column not in df.columns:
            raise ValueError(
                f"Text column '{self.config.text_column}' not found in dataset"
            )
        if self.config.label_column not in df.columns:
            raise ValueError(
                f"Label column '{self.config.label_column}' not found in dataset"
            )

        # Handle missing values
        print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
        df = df.dropna(subset=[self.config.text_column, self.config.label_column])

        # Combine title and text if specified
        if self.config.combine_title_text and self.config.title_column in df.columns:
            print("Combining title and text...")
            df[self.config.text_column] = (
                df[self.config.title_column].fillna("")
                + " "
                + df[self.config.text_column]
            )

        # Clean text
        df[self.config.text_column] = (
            df[self.config.text_column].astype(str).str.strip()
        )

        # Ensure binary labels (0 = fake, 1 = real)
        unique_labels = df[self.config.label_column].unique()
        print(f"Unique labels found: {unique_labels}")

        # Convert labels to integers if needed
        df[self.config.label_column] = 1 - df[self.config.label_column]
        df[self.config.label_column] = df[self.config.label_column].astype(int)

        # Validate labels
        if not set(df[self.config.label_column].unique()).issubset({0, 1}):
            raise ValueError("Labels must be 0 (fake) or 1 (real)")

        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates(subset=[self.config.text_column])
        print(f"Removed {initial_len - len(df)} duplicate texts")

        # Remove empty texts
        df = df[df[self.config.text_column].str.len() > 0]

        # Apply sampling if sample_ratio is specified (for quick runs)
        if self.config.sample_ratio is not None:
            if not 0 < self.config.sample_ratio <= 1:
                raise ValueError("sample_ratio must be between 0 and 1")
            print(
                f"\n⚡ QUICK RUN MODE: Using {self.config.sample_ratio * 100:.0f}% of data"
            )
            df = df.groupby(self.config.label_column, group_keys=False).apply(
                lambda x: x.sample(
                    frac=self.config.sample_ratio, random_state=self.config.random_state
                )
            )
            print(f"Sampled dataset shape: {df.shape}")

        # Display label distribution
        label_counts = df[self.config.label_column].value_counts()
        print("Label distribution:")
        print(
            f"  Fake (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0) / len(df) * 100:.1f}%)"
        )
        print(
            f"  Real (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0) / len(df) * 100:.1f}%)"
        )
        print(f"Final dataset shape: {df.shape}")

        self.df = df
        return df

    def create_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train, validation, and test splits

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.df is None:
            self.load_data()

        print("Creating data splits...")

        X = self.df[self.config.text_column].values
        y = self.df[self.config.label_column].values

        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Second split: train and val
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp,
        )

        # Create DataFrames
        train_df = pd.DataFrame({"text": X_train, "label": y_train})
        val_df = pd.DataFrame({"text": X_val, "label": y_val})
        test_df = pd.DataFrame({"text": X_test, "label": y_test})

        print(f"Split sizes:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")

        return train_df, val_df, test_df

    def create_datasets(self) -> DatasetDict:
        """
        Create HuggingFace datasets

        Returns:
            DatasetDict with train, validation, and test splits
        """
        train_df, val_df, test_df = self.create_splits()

        # Convert to HuggingFace Dataset format
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

        # Create DatasetDict
        self.datasets = DatasetDict(
            {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        )

        return self.datasets

    def tokenize_datasets(
        self, tokenizer: DistilBertTokenizer, max_length: int
    ) -> DatasetDict:
        """
        Tokenize the datasets

        Args:
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length

        Returns:
            Tokenized DatasetDict
        """
        if self.datasets is None:
            self.create_datasets()

        self.tokenizer = tokenizer

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

        print("Tokenizing datasets...")
        tokenized_datasets = self.datasets.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        return tokenized_datasets

    def get_sample_texts(self, n: int = 5) -> list:
        """Get sample texts from the dataset"""
        if self.df is None:
            self.load_data()

        samples = []
        for label in [0, 1]:
            label_df = self.df[self.df[self.config.label_column] == label]
            sample_df = label_df.sample(
                min(n, len(label_df)), random_state=self.config.random_state
            )
            for _, row in sample_df.iterrows():
                samples.append(
                    {
                        "text": row[self.config.text_column][:200] + "..."
                        if len(row[self.config.text_column]) > 200
                        else row[self.config.text_column],
                        "label": "Real"
                        if row[self.config.label_column] == 1
                        else "Fake",
                    }
                )
        return samples
