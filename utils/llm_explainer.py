"""
LLM-based explanation generation for federated learning processes.
Uses OpenAI API to generate natural language explanations.
"""
from openai import OpenAI
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class LLMExplainer:
    """Generates explanations using OpenAI's LLM."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM explainer.

        Args:
            api_key: OpenAI API key.
            model: Model to use for generation.
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"LLM Explainer initialized with model: {model}")

    def generate_explanation(self, prompt: str, max_tokens: int = 1500) -> str:
        """
        Generate an explanation using the LLM.

        Args:
            prompt: Prompt for the LLM.
            max_tokens: Maximum tokens in the response.

        Returns:
            Generated explanation text.
        """
        try:
            logger.info("Generating explanation with LLM...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in federated learning, machine learning, and data privacy. Explain concepts clearly and concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=max_tokens
            )

            explanation = response.choices[0].message.content.strip()
            logger.info("Explanation generated successfully.")
            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            raise

    def explain_federated_learning_process(
        self,
        data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        models: Dict,
        global_model,
        include_weights: bool = False
    ) -> str:
        """
        Generate a comprehensive explanation of the federated learning process.

        Args:
            data_dict: Dictionary of bank data.
            models: Dictionary of trained local models.
            global_model: The trained global model.
            include_weights: Whether to include detailed weight information.

        Returns:
            Explanation text.
        """
        logger.info("Preparing federated learning explanation...")

        # Prepare data summary
        data_summary = self._create_data_summary(data_dict)

        # Prepare model summary
        model_summary = self._create_model_summary(models, include_weights)

        # Prepare global model summary
        global_summary = self._create_global_model_summary(global_model, include_weights)

        # Create the prompt
        prompt = f"""
The following is a summary of a federated learning simulation for fraud detection:

## Synthetic Data Created:
{data_summary}

## Local Models Trained:
{model_summary}

## Encryption and Aggregation:
Local model weights were encrypted using homomorphic encryption (CKKS scheme from TenSEAL).
Encrypted weights were aggregated while still encrypted (federated averaging).
The aggregated weights were then decrypted to create the global model.

## Global Model:
{global_summary}

## Task:
Please provide a clear, educational explanation of this federated learning process. Structure your response as follows:

1. **Overview**: Brief introduction to what happened (2-3 sentences)

2. **Data Generation**: Explain the fraud data created for each bank, including:
   - Number of transactions per bank
   - Fraud rates for each bank
   - Why banks have different fraud patterns

3. **Local Training**: Explain what happened during local model training:
   - Each bank trained on their own data
   - Why this preserves privacy
   - What the models learned

4. **Homomorphic Encryption**: Explain:
   - What homomorphic encryption is (in simple terms)
   - Why it's important for federated learning
   - How it enables secure aggregation

5. **Federated Aggregation**: Explain:
   - How the encrypted weights were combined
   - Why averaging works
   - The result: a global model

6. **Benefits & Conclusion**: Explain:
   - Why federated learning is valuable in this context
   - Key privacy benefits
   - When this approach is useful

Use clear language, bullet points where appropriate, and focus on educational value.
"""

        return self.generate_explanation(prompt)

    def _create_data_summary(self, data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> str:
        """Create a summary of the data."""
        summary_parts = []

        for bank_name, (X, y) in data_dict.items():
            fraud_count = np.sum(y)
            total = len(y)
            fraud_rate = (fraud_count / total) * 100

            summary_parts.append(
                f"- {bank_name}: {total} transactions, {fraud_count} fraudulent ({fraud_rate:.1f}%)"
            )

        return "\n".join(summary_parts)

    def _create_model_summary(self, models: Dict, include_weights: bool = False) -> str:
        """Create a summary of the trained models."""
        summary_parts = []

        for bank_name, model in models.items():
            summary_parts.append(f"- {bank_name}: Model trained with {len(model.layers)} layers")

            if include_weights:
                weights = model.get_weights()
                for idx, w in enumerate(weights):
                    summary_parts.append(f"  Layer {idx}: shape {w.shape}, mean={np.mean(w):.4f}")

        return "\n".join(summary_parts)

    def _create_global_model_summary(self, global_model, include_weights: bool = False) -> str:
        """Create a summary of the global model."""
        summary = f"Global model created with {len(global_model.layers)} layers"

        if include_weights:
            weights = global_model.get_weights()
            weight_info = []
            for idx, w in enumerate(weights):
                weight_info.append(f"  Layer {idx}: shape {w.shape}, mean={np.mean(w):.4f}, std={np.std(w):.4f}")
            summary += "\n" + "\n".join(weight_info)

        return summary

    def explain_encryption_process(self, encryption_info: Dict) -> str:
        """
        Generate an explanation of the encryption process.

        Args:
            encryption_info: Dictionary with encryption information.

        Returns:
            Explanation text.
        """
        prompt = f"""
Explain the homomorphic encryption process used in this federated learning system:

- Number of layers encrypted: {encryption_info['num_layers']}
- Total encrypted data size: {encryption_info['total_encrypted_size']:,} bytes
- Encryption scheme: CKKS (from TenSEAL library)

Provide a brief, clear explanation of:
1. What homomorphic encryption means
2. Why CKKS is suitable for machine learning
3. The security guarantees it provides
4. The computational overhead involved

Keep it concise (3-4 paragraphs).
"""

        return self.generate_explanation(prompt, max_tokens=500)
