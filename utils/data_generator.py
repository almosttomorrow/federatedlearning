"""
Synthetic data generation for federated learning simulation.
Creates fraud detection datasets for multiple banks.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DataGenerator:
    """Generates synthetic fraud detection data for multiple banks."""

    def __init__(self, num_banks: int = 3, num_samples: int = 1000,
                 num_features: int = 10, random_seed: int = 42):
        """
        Initialize the data generator.

        Args:
            num_banks: Number of banks to generate data for.
            num_samples: Number of samples per bank.
            num_features: Number of features in the dataset.
            random_seed: Random seed for reproducibility.
        """
        self.num_banks = num_banks
        self.num_samples = num_samples
        self.num_features = num_features
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def create_synthetic_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Create synthetic fraud data for all banks.

        Returns:
            Dict mapping bank names to (features, labels) tuples.
        """
        logger.info(f"Generating synthetic data for {self.num_banks} banks...")
        data_dict = {}

        for bank_idx in range(self.num_banks):
            bank_name = f'Bank_{bank_idx + 1}'

            # Generate features with slight variations per bank
            X = np.random.randn(self.num_samples, self.num_features) + (bank_idx * 0.1)

            # Generate labels with realistic fraud rate (5-15%)
            fraud_rate = 0.05 + (bank_idx * 0.05)
            y = (np.random.random(self.num_samples) < fraud_rate).astype(int).reshape(-1, 1)

            data_dict[bank_name] = (X, y)
            fraud_count = np.sum(y)
            logger.info(f"{bank_name}: {fraud_count}/{self.num_samples} fraud cases ({fraud_rate*100:.1f}%)")

        return data_dict

    def save_data(self, data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
                  output_dir: str = '.') -> None:
        """
        Save generated data to CSV files.

        Args:
            data_dict: Dictionary of bank data.
            output_dir: Directory to save CSV files.
        """
        logger.info(f"Saving data to {output_dir}...")

        for bank_name, (X, y) in data_dict.items():
            # Save features
            feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_cols)
            X_df.to_csv(f'{output_dir}/{bank_name}_features.csv', index=False)

            # Save labels
            y_df = pd.DataFrame(y, columns=['label'])
            y_df.to_csv(f'{output_dir}/{bank_name}_labels.csv', index=False)

        logger.info("Data saved successfully.")

    def get_data_summary(self, data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> str:
        """
        Get a summary of the generated data.

        Args:
            data_dict: Dictionary of bank data.

        Returns:
            str: Data summary.
        """
        summary = []
        for bank_name, (X, y) in data_dict.items():
            fraud_count = np.sum(y)
            total = len(y)
            fraud_rate = (fraud_count / total) * 100

            summary.append(
                f"{bank_name}:\n"
                f"  - Total transactions: {total}\n"
                f"  - Fraud cases: {fraud_count} ({fraud_rate:.1f}%)\n"
                f"  - Feature means: {np.mean(X, axis=0).round(2)}\n"
                f"  - Feature stds: {np.std(X, axis=0).round(2)}"
            )

        return "\n\n".join(summary)
