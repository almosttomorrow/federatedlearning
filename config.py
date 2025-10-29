"""
Configuration management for the Federated Learning MVP.
Handles environment variables and application settings.
"""
import os
from typing import Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Application configuration loaded from environment variables."""

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

    # Model Configuration
    NUM_BANKS: int = int(os.getenv('NUM_BANKS', '3'))
    NUM_SAMPLES: int = int(os.getenv('NUM_SAMPLES', '1000'))
    NUM_FEATURES: int = int(os.getenv('NUM_FEATURES', '10'))
    EPOCHS: int = int(os.getenv('EPOCHS', '10'))
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '32'))

    # TenSEAL Encryption Configuration
    POLY_MODULUS_DEGREE: int = int(os.getenv('POLY_MODULUS_DEGREE', '8192'))
    COEFF_MOD_BIT_SIZES: list = [60, 40, 40, 60]
    GLOBAL_SCALE: int = 2**40

    # Random Seed for Reproducibility
    RANDOM_SEED: int = int(os.getenv('RANDOM_SEED', '42'))

    @classmethod
    def validate(cls) -> bool:
        """
        Validate required configuration.

        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        if not cls.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set. LLM explanations will be disabled.")
            return False
        return True

    @classmethod
    def get_summary(cls) -> str:
        """
        Get a summary of current configuration.

        Returns:
            str: Configuration summary.
        """
        return f"""
        Configuration:
        - Number of Banks: {cls.NUM_BANKS}
        - Samples per Bank: {cls.NUM_SAMPLES}
        - Features: {cls.NUM_FEATURES}
        - Training Epochs: {cls.EPOCHS}
        - Batch Size: {cls.BATCH_SIZE}
        - OpenAI Model: {cls.OPENAI_MODEL}
        - API Key Set: {'Yes' if cls.OPENAI_API_KEY else 'No'}
        """
