"""
Homomorphic encryption utilities for federated learning.
Uses TenSEAL for CKKS encryption scheme.
"""
import tenseal as ts
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class EncryptionManager:
    """Manages homomorphic encryption operations using TenSEAL."""

    def __init__(self, poly_modulus_degree: int = 8192,
                 coeff_mod_bit_sizes: List[int] = None,
                 global_scale: int = 2**40):
        """
        Initialize the encryption manager.

        Args:
            poly_modulus_degree: Polynomial modulus degree for CKKS.
            coeff_mod_bit_sizes: Coefficient modulus bit sizes.
            global_scale: Global scale for CKKS encoding.
        """
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]

        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = global_scale
        self.context = self._initialize_context()

    def _initialize_context(self) -> ts.Context:
        """
        Initialize TenSEAL context for homomorphic encryption.

        Returns:
            TenSEAL context object.
        """
        try:
            logger.info("Initializing TenSEAL context...")
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=self.coeff_mod_bit_sizes
            )
            context.generate_galois_keys()
            context.global_scale = self.global_scale
            logger.info("TenSEAL context initialized successfully.")
            return context
        except Exception as e:
            logger.error(f"Error initializing TenSEAL context: {e}")
            raise

    def encrypt_weights(self, weights: List[np.ndarray]) -> List[ts.CKKSVector]:
        """
        Encrypt model weights using homomorphic encryption.

        Args:
            weights: List of weight arrays from a model.

        Returns:
            List of encrypted weight vectors.
        """
        try:
            logger.info("Encrypting model weights...")
            encrypted_weights = []

            for idx, layer_weights in enumerate(weights):
                # Flatten the layer weights
                flat_weights = layer_weights.flatten().tolist()

                # Encrypt the flattened weights
                encrypted_layer = ts.ckks_vector(self.context, flat_weights)
                encrypted_weights.append(encrypted_layer)

                logger.debug(f"Layer {idx} encrypted: shape {layer_weights.shape}")

            logger.info(f"Successfully encrypted {len(encrypted_weights)} layers.")
            return encrypted_weights

        except Exception as e:
            logger.error(f"Error encrypting weights: {e}")
            raise

    def decrypt_weights(self, encrypted_weights: List[ts.CKKSVector]) -> List[np.ndarray]:
        """
        Decrypt encrypted model weights.

        Args:
            encrypted_weights: List of encrypted weight vectors.

        Returns:
            List of decrypted weight arrays (flattened).
        """
        try:
            logger.info("Decrypting model weights...")
            decrypted_weights = []

            for idx, encrypted_layer in enumerate(encrypted_weights):
                # Decrypt the layer
                decrypted_layer = np.array(encrypted_layer.decrypt())
                decrypted_weights.append(decrypted_layer)
                logger.debug(f"Layer {idx} decrypted: {len(decrypted_layer)} values")

            logger.info(f"Successfully decrypted {len(decrypted_weights)} layers.")
            return decrypted_weights

        except Exception as e:
            logger.error(f"Error decrypting weights: {e}")
            raise

    def aggregate_encrypted_weights(self,
                                    encrypted_weights_dict: Dict[str, List[ts.CKKSVector]]) -> List[ts.CKKSVector]:
        """
        Aggregate encrypted weights from multiple banks using federated averaging.

        Args:
            encrypted_weights_dict: Dictionary mapping bank names to encrypted weights.

        Returns:
            List of aggregated encrypted weight vectors.
        """
        try:
            logger.info("Aggregating encrypted weights...")
            num_banks = len(encrypted_weights_dict)

            # Get the first bank's weights to determine structure
            first_bank_weights = list(encrypted_weights_dict.values())[0]
            num_layers = len(first_bank_weights)

            aggregated_weights = []

            # Aggregate layer by layer
            for layer_idx in range(num_layers):
                # Start with zeros
                layer_sum = None

                for bank_name, encrypted_weights in encrypted_weights_dict.items():
                    encrypted_layer = encrypted_weights[layer_idx]

                    if layer_sum is None:
                        # First bank - just copy
                        layer_sum = encrypted_layer
                    else:
                        # Add to sum (homomorphic addition)
                        layer_sum = layer_sum + encrypted_layer

                # Average by dividing by number of banks
                layer_avg = layer_sum * (1.0 / num_banks)
                aggregated_weights.append(layer_avg)

                logger.debug(f"Layer {layer_idx} aggregated from {num_banks} banks")

            logger.info(f"Successfully aggregated {num_layers} layers from {num_banks} banks.")
            return aggregated_weights

        except Exception as e:
            logger.error(f"Error aggregating encrypted weights: {e}")
            raise

    def reshape_decrypted_weights(self, decrypted_weights: List[np.ndarray],
                                  reference_model) -> List[np.ndarray]:
        """
        Reshape decrypted weights to match the model's layer shapes.

        Args:
            decrypted_weights: List of flattened decrypted weight arrays.
            reference_model: Reference model to get layer shapes from.

        Returns:
            List of reshaped weight arrays.
        """
        try:
            logger.info("Reshaping decrypted weights...")
            layer_shapes = [layer.shape for layer in reference_model.get_weights()]
            reshaped_weights = []

            for idx, (flat_weights, target_shape) in enumerate(zip(decrypted_weights, layer_shapes)):
                # Calculate expected size
                expected_size = np.prod(target_shape)

                # Take only the required number of values
                reshaped = flat_weights[:expected_size].reshape(target_shape)
                reshaped_weights.append(reshaped)

                logger.debug(f"Layer {idx} reshaped to {target_shape}")

            logger.info("Weights reshaped successfully.")
            return reshaped_weights

        except Exception as e:
            logger.error(f"Error reshaping weights: {e}")
            raise

    def get_encryption_info(self, encrypted_weights: List[ts.CKKSVector]) -> Dict:
        """
        Get information about encrypted weights.

        Args:
            encrypted_weights: List of encrypted weight vectors.

        Returns:
            Dictionary with encryption information.
        """
        info = {
            'num_layers': len(encrypted_weights),
            'serialized_sizes': [len(layer.serialize()) for layer in encrypted_weights],
            'total_encrypted_size': sum(len(layer.serialize()) for layer in encrypted_weights)
        }
        return info
