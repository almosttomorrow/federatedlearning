"""
Model training utilities for federated learning.
Handles model creation, training, and evaluation.
"""
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, List
import logging
import pickle

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model creation and training for federated learning."""

    def __init__(self, num_features: int = 10, epochs: int = 10,
                 batch_size: int = 32, verbose: int = 0):
        """
        Initialize the model trainer.

        Args:
            num_features: Number of input features.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            verbose: Verbosity level for training.
        """
        self.num_features = num_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def create_model(self) -> tf.keras.Model:
        """
        Create a Keras model for binary classification.

        Returns:
            Compiled Keras model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu',
                                input_shape=(self.num_features,),
                                name='hidden_layer'),
            tf.keras.layers.Dropout(0.2, name='dropout'),
            tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        return model

    def train_local_models(self, data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, tf.keras.Model]:
        """
        Train local models for each bank.

        Args:
            data_dict: Dictionary mapping bank names to (features, labels).

        Returns:
            Dictionary mapping bank names to trained models.
        """
        logger.info("Training local models for each bank...")
        models = {}

        for bank_name, (X_train, y_train) in data_dict.items():
            logger.info(f"Training model for {bank_name}...")

            model = self.create_model()

            # Train the model
            history = model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                validation_split=0.2
            )

            # Log final metrics
            final_loss = history.history['loss'][-1]
            final_acc = history.history['accuracy'][-1]
            logger.info(f"{bank_name} - Final Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")

            models[bank_name] = model

        logger.info("All local models trained successfully.")
        return models

    def save_model_weights(self, model: tf.keras.Model, filepath: str) -> None:
        """
        Save model weights to a file.

        Args:
            model: Keras model.
            filepath: Path to save weights.
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model.get_weights(), f)
            logger.info(f"Model weights saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model weights: {e}")
            raise

    def load_model_weights(self, filepath: str) -> List[np.ndarray]:
        """
        Load model weights from a file.

        Args:
            filepath: Path to load weights from.

        Returns:
            List of weight arrays.
        """
        try:
            with open(filepath, 'rb') as f:
                weights = pickle.load(f)
            logger.info(f"Model weights loaded from {filepath}")
            return weights
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            raise

    def aggregate_weights(self, models: Dict[str, tf.keras.Model]) -> List[np.ndarray]:
        """
        Aggregate weights from multiple models using federated averaging.

        Args:
            models: Dictionary of trained models.

        Returns:
            List of averaged weight arrays.
        """
        logger.info("Aggregating model weights...")

        # Get all model weights
        all_weights = [model.get_weights() for model in models.values()]

        # Average weights layer by layer
        aggregated_weights = []
        for layer_idx in range(len(all_weights[0])):
            layer_weights = [weights[layer_idx] for weights in all_weights]
            avg_layer_weights = np.mean(layer_weights, axis=0)
            aggregated_weights.append(avg_layer_weights)

        logger.info("Weights aggregated successfully.")
        return aggregated_weights

    def create_global_model(self, aggregated_weights: List[np.ndarray]) -> tf.keras.Model:
        """
        Create a global model with aggregated weights.

        Args:
            aggregated_weights: List of aggregated weight arrays.

        Returns:
            Global model with aggregated weights.
        """
        logger.info("Creating global model...")
        global_model = self.create_model()
        global_model.set_weights(aggregated_weights)
        logger.info("Global model created successfully.")
        return global_model

    def evaluate_model(self, model: tf.keras.Model, X: np.ndarray,
                       y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a model on test data.

        Args:
            model: Keras model to evaluate.
            X: Test features.
            y: Test labels.

        Returns:
            Dictionary of evaluation metrics.
        """
        results = model.evaluate(X, y, verbose=0)
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'auc': results[2] if len(results) > 2 else None
        }
        return metrics
