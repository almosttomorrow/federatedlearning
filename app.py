"""
Federated Learning MVP with Homomorphic Encryption and LLM Explanations.

This Streamlit application demonstrates federated learning with:
- Synthetic fraud detection data for multiple banks
- Local model training at each bank
- Homomorphic encryption for secure weight sharing
- Federated averaging for global model creation
- LLM-powered explanations of the process
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils import DataGenerator, ModelTrainer, EncryptionManager, LLMExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Federated Learning MVP",
    page_icon="🔐",
    layout="wide"
)


# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'weights_encrypted' not in st.session_state:
        st.session_state.weights_encrypted = False
    if 'global_model_created' not in st.session_state:
        st.session_state.global_model_created = False


def plot_feature_distributions(data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], bank_name: str):
    """Plot feature distributions for a bank."""
    X, y = data_dict[bank_name]

    # Create subplots for features
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(min(10, X.shape[1])):
        sns.histplot(X[:, i], bins=30, ax=axes[i], color=sns.color_palette("husl", 10)[i], kde=True)
        axes[i].set_title(f'Feature {i+1}', fontsize=10)
        axes[i].set_xlabel('')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Plot label distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    fraud_count = np.sum(y)
    no_fraud_count = len(y) - fraud_count

    ax.bar(['No Fraud', 'Fraud'], [no_fraud_count, fraud_count], color=['#2ecc71', '#e74c3c'])
    ax.set_ylabel('Count')
    ax.set_title('Transaction Label Distribution')

    for i, v in enumerate([no_fraud_count, fraud_count]):
        ax.text(i, v + 10, str(int(v)), ha='center', va='bottom', fontweight='bold')

    st.pyplot(fig)
    plt.close()


def plot_model_weights(model, title="Model Weights"):
    """Visualize model weights as heatmaps."""
    weights = model.get_weights()

    # Plot only weight matrices (not biases)
    weight_matrices = [w for w in weights if len(w.shape) == 2]

    if not weight_matrices:
        st.warning("No 2D weight matrices to visualize.")
        return

    fig, axes = plt.subplots(1, len(weight_matrices), figsize=(6 * len(weight_matrices), 5))

    if len(weight_matrices) == 1:
        axes = [axes]

    for idx, weight_matrix in enumerate(weight_matrices):
        sns.heatmap(weight_matrix, ax=axes[idx], cmap='coolwarm', center=0, cbar=True)
        axes[idx].set_title(f'Layer {idx + 1} Weights')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">🔐 Federated Learning MVP</div>', unsafe_allow_html=True)
    st.markdown("""
    This application demonstrates **federated learning** with **homomorphic encryption** for privacy-preserving
    collaborative machine learning in fraud detection.
    """)

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Check if API key is configured
        api_key_configured = Config.OPENAI_API_KEY is not None and Config.OPENAI_API_KEY != ""

        if api_key_configured:
            st.success("✓ OpenAI API Key Configured")
        else:
            st.warning("⚠️ OpenAI API Key Not Set")
            st.info("Set OPENAI_API_KEY environment variable to enable LLM explanations.")

        st.markdown("---")

        # Configuration parameters
        num_banks = st.slider("Number of Banks", 2, 5, Config.NUM_BANKS)
        num_samples = st.slider("Samples per Bank", 500, 2000, Config.NUM_SAMPLES, step=100)
        epochs = st.slider("Training Epochs", 5, 20, Config.EPOCHS)

        st.markdown("---")

        # Show current config
        with st.expander("View Configuration"):
            st.code(Config.get_summary())

    # Main workflow
    st.markdown('<div class="sub-header">Step 1: Generate Synthetic Data</div>', unsafe_allow_html=True)

    st.markdown("""
    Generate synthetic fraud detection data for multiple banks. Each bank has:
    - Different transaction patterns
    - Varying fraud rates (5-15%)
    - Private data that won't be shared directly
    """)

    if st.button("🎲 Generate Synthetic Data", key="generate_data"):
        with st.spinner("Generating synthetic data..."):
            try:
                # Create data generator
                data_gen = DataGenerator(
                    num_banks=num_banks,
                    num_samples=num_samples,
                    num_features=Config.NUM_FEATURES,
                    random_seed=Config.RANDOM_SEED
                )

                # Generate data
                data_dict = data_gen.create_synthetic_data()

                # Save to session state
                st.session_state.data_dict = data_dict
                st.session_state.data_generated = True

                # Save to CSV
                data_gen.save_data(data_dict)

                st.markdown('<div class="success-box">✅ Synthetic data generated successfully!</div>', unsafe_allow_html=True)

                # Show data summary
                summary = data_gen.get_data_summary(data_dict)
                with st.expander("📊 View Data Summary", expanded=True):
                    st.text(summary)

            except Exception as e:
                st.error(f"Error generating data: {e}")
                logger.error(f"Data generation error: {e}", exc_info=True)

    # Display data visualizations if data exists
    if st.session_state.data_generated:
        st.markdown("### 📈 Data Visualizations")

        for bank_name in st.session_state.data_dict.keys():
            with st.expander(f"View {bank_name} Data"):
                plot_feature_distributions(st.session_state.data_dict, bank_name)

    # Step 2: Train Local Models
    if st.session_state.data_generated:
        st.markdown('<div class="sub-header">Step 2: Train Local Models</div>', unsafe_allow_html=True)

        st.markdown("""
        Each bank trains a model on their private data locally.
        No raw data is shared between banks.
        """)

        if st.button("🤖 Train Local Models", key="train_models"):
            with st.spinner("Training local models..."):
                try:
                    # Create model trainer
                    trainer = ModelTrainer(
                        num_features=Config.NUM_FEATURES,
                        epochs=epochs,
                        batch_size=Config.BATCH_SIZE,
                        verbose=0
                    )

                    # Train models
                    models = trainer.train_local_models(st.session_state.data_dict)

                    # Save to session state
                    st.session_state.models = models
                    st.session_state.models_trained = True
                    st.session_state.trainer = trainer

                    st.markdown('<div class="success-box">✅ All local models trained successfully!</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error training models: {e}")
                    logger.error(f"Model training error: {e}", exc_info=True)

    # Display model weights if trained
    if st.session_state.models_trained:
        st.markdown("### 🧠 Model Weights Visualization")

        for bank_name, model in st.session_state.models.items():
            with st.expander(f"View {bank_name} Model Weights"):
                plot_model_weights(model, title=f"{bank_name} Model Weights")

    # Step 3: Encrypt Weights
    if st.session_state.models_trained:
        st.markdown('<div class="sub-header">Step 3: Encrypt Model Weights</div>', unsafe_allow_html=True)

        st.markdown("""
        Model weights are encrypted using **homomorphic encryption** (CKKS scheme).
        This allows secure aggregation without revealing individual bank's model parameters.
        """)

        if st.button("🔒 Encrypt Weights", key="encrypt_weights"):
            with st.spinner("Encrypting model weights..."):
                try:
                    # Initialize encryption manager
                    enc_manager = EncryptionManager(
                        poly_modulus_degree=Config.POLY_MODULUS_DEGREE,
                        coeff_mod_bit_sizes=Config.COEFF_MOD_BIT_SIZES,
                        global_scale=Config.GLOBAL_SCALE
                    )

                    # Encrypt weights for each bank
                    encrypted_weights_dict = {}
                    encryption_info = {}

                    for bank_name, model in st.session_state.models.items():
                        encrypted_weights = enc_manager.encrypt_weights(model.get_weights())
                        encrypted_weights_dict[bank_name] = encrypted_weights
                        encryption_info[bank_name] = enc_manager.get_encryption_info(encrypted_weights)

                    # Save to session state
                    st.session_state.encrypted_weights_dict = encrypted_weights_dict
                    st.session_state.enc_manager = enc_manager
                    st.session_state.encryption_info = encryption_info
                    st.session_state.weights_encrypted = True

                    st.markdown('<div class="success-box">✅ Model weights encrypted successfully!</div>', unsafe_allow_html=True)

                    # Show encryption info
                    with st.expander("🔐 View Encryption Information"):
                        for bank_name, info in encryption_info.items():
                            st.write(f"**{bank_name}:**")
                            st.write(f"  - Encrypted layers: {info['num_layers']}")
                            st.write(f"  - Total size: {info['total_encrypted_size']:,} bytes")

                except Exception as e:
                    st.error(f"Error encrypting weights: {e}")
                    logger.error(f"Encryption error: {e}", exc_info=True)

    # Step 4: Federated Aggregation
    if st.session_state.weights_encrypted:
        st.markdown('<div class="sub-header">Step 4: Federated Aggregation</div>', unsafe_allow_html=True)

        st.markdown("""
        Encrypted weights from all banks are aggregated using **federated averaging**.
        The aggregation happens on encrypted data, preserving privacy.
        """)

        if st.button("🌐 Aggregate & Create Global Model", key="aggregate"):
            with st.spinner("Aggregating encrypted weights and creating global model..."):
                try:
                    enc_manager = st.session_state.enc_manager
                    trainer = st.session_state.trainer

                    # Aggregate encrypted weights
                    aggregated_encrypted = enc_manager.aggregate_encrypted_weights(
                        st.session_state.encrypted_weights_dict
                    )

                    # Decrypt aggregated weights
                    decrypted_aggregated = enc_manager.decrypt_weights(aggregated_encrypted)

                    # Create reference model for reshaping
                    reference_model = trainer.create_model()

                    # Reshape weights
                    reshaped_weights = enc_manager.reshape_decrypted_weights(
                        decrypted_aggregated,
                        reference_model
                    )

                    # Create global model
                    global_model = trainer.create_global_model(reshaped_weights)

                    # Save to session state
                    st.session_state.global_model = global_model
                    st.session_state.global_model_created = True

                    # Save global model weights
                    trainer.save_model_weights(global_model, 'global_model_weights.pkl')

                    st.markdown('<div class="success-box">✅ Global model created successfully!</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error creating global model: {e}")
                    logger.error(f"Global model creation error: {e}", exc_info=True)

    # Display global model
    if st.session_state.global_model_created:
        st.markdown("### 🌍 Global Model Weights")
        plot_model_weights(st.session_state.global_model, title="Global Model Weights")

        # Download button
        try:
            with open('global_model_weights.pkl', 'rb') as f:
                st.download_button(
                    label="📥 Download Global Model",
                    data=f,
                    file_name="global_model_weights.pkl",
                    mime="application/octet-stream"
                )
        except:
            pass

    # Step 5: Generate Explanation
    if st.session_state.global_model_created:
        st.markdown('<div class="sub-header">Step 5: Generate LLM Explanation</div>', unsafe_allow_html=True)

        st.markdown("""
        Use an LLM to generate a comprehensive explanation of the entire federated learning process.
        """)

        if not api_key_configured:
            st.markdown('<div class="warning-box">⚠️ OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable to use this feature.</div>', unsafe_allow_html=True)
        else:
            if st.button("💡 Generate Explanation", key="explain"):
                with st.spinner("Generating explanation with LLM..."):
                    try:
                        # Initialize LLM explainer
                        explainer = LLMExplainer(
                            api_key=Config.OPENAI_API_KEY,
                            model=Config.OPENAI_MODEL
                        )

                        # Generate explanation
                        explanation = explainer.explain_federated_learning_process(
                            data_dict=st.session_state.data_dict,
                            models=st.session_state.models,
                            global_model=st.session_state.global_model,
                            include_weights=False
                        )

                        st.markdown('<div class="success-box">✅ Explanation generated!</div>', unsafe_allow_html=True)

                        # Display explanation
                        st.markdown("### 📝 Federated Learning Process Explanation")
                        st.markdown(explanation)

                    except Exception as e:
                        st.error(f"Error generating explanation: {e}")
                        logger.error(f"Explanation generation error: {e}", exc_info=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with Streamlit, TensorFlow, TenSEAL, and OpenAI</p>
        <p>Federated Learning MVP - Privacy-Preserving Collaborative ML</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
