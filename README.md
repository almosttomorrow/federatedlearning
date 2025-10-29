# Federated Learning MVP with Homomorphic Encryption

A production-ready prototype demonstrating privacy-preserving federated learning for fraud detection using homomorphic encryption and LLM-powered explanations.

## Overview

This application showcases how multiple financial institutions (banks) can collaboratively train a machine learning model for fraud detection without sharing their sensitive transaction data. The system uses:

- **Federated Learning**: Distributed model training where data stays local
- **Homomorphic Encryption**: Secure weight aggregation using TenSEAL (CKKS scheme)
- **LLM Explanations**: Natural language explanations of the learning process
- **Interactive UI**: Streamlit-based web interface for step-by-step demonstration

## Key Features

- **Privacy-Preserving**: Raw data never leaves individual banks
- **Secure Aggregation**: Model weights are encrypted during transmission and aggregation
- **Modular Architecture**: Clean separation of concerns with well-documented components
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Configurable**: Environment-based configuration for easy customization
- **Educational**: Built-in LLM explanations for understanding the process

## Architecture

```
federatedlearning/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration management
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── data_generator.py      # Synthetic data generation
│   ├── model_trainer.py       # Model training and aggregation
│   ├── encryption.py          # Homomorphic encryption operations
│   └── llm_explainer.py       # LLM-powered explanations
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/almosttomorrow/federatedlearning.git
cd federatedlearning
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: TenSEAL installation may require additional build tools on some systems. See [TenSEAL installation guide](https://github.com/OpenMined/TenSEAL#installation) if you encounter issues.

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
# At minimum, set your OpenAI API key:
# OPENAI_API_KEY=your_actual_api_key_here
```

Get your OpenAI API key from: https://platform.openai.com/api-keys

**Note**: The LLM explanation feature is optional. The app will work without an API key, but explanations will be disabled.

## Configuration

All configuration is managed through environment variables. See `.env.example` for available options:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM explanations | None |
| `OPENAI_MODEL` | OpenAI model to use | gpt-3.5-turbo |
| `NUM_BANKS` | Number of banks to simulate | 3 |
| `NUM_SAMPLES` | Samples per bank | 1000 |
| `NUM_FEATURES` | Number of features in dataset | 10 |
| `EPOCHS` | Training epochs | 10 |
| `BATCH_SIZE` | Training batch size | 32 |
| `RANDOM_SEED` | Random seed for reproducibility | 42 |

## Usage

### Option 1: Running Locally

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Option 2: Running in Google Colab

**🚀 Quick Start:** Open the pre-made notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/almosttomorrow/federatedlearning/blob/main/Federated_Learning_Colab.ipynb)

**Or follow the manual setup:**

See [COLAB_SETUP.md](COLAB_SETUP.md) for detailed step-by-step instructions.

**Quick Summary:**
1. Open the notebook in Colab
2. Install dependencies: `!pip install streamlit tensorflow numpy pandas matplotlib seaborn tenseal openai python-dotenv pyngrok`
3. Clone repo: `!git clone https://github.com/almosttomorrow/federatedlearning.git`
4. Set API key (optional) using the code in the notebook
5. Run the app with ngrok tunnel
6. Click the public URL to access the app

**Benefits of Colab:**
- No local installation required
- Free GPU/TPU access (though CPU is sufficient)
- Pre-configured environment
- Easy sharing and collaboration

### Workflow Steps

The application guides you through a 5-step federated learning workflow:

#### Step 1: Generate Synthetic Data
- Creates fraud detection datasets for multiple banks
- Each bank has different transaction patterns and fraud rates
- Data is visualized with feature distributions and fraud statistics

#### Step 2: Train Local Models
- Each bank trains a neural network on their local data
- Models are trained independently without sharing data
- Model weights are visualized for each bank

#### Step 3: Encrypt Model Weights
- Model weights are encrypted using homomorphic encryption
- Uses TenSEAL's CKKS scheme for floating-point operations
- Encryption information is displayed

#### Step 4: Federated Aggregation
- Encrypted weights are aggregated using federated averaging
- Aggregation happens on encrypted data
- Global model is created from decrypted aggregated weights
- Global model weights can be downloaded

#### Step 5: Generate LLM Explanation
- OpenAI's LLM generates a comprehensive explanation
- Explains the entire process in educational terms
- Requires OpenAI API key

### Customization

You can customize the simulation using the sidebar controls:
- **Number of Banks**: Adjust the number of participating institutions (2-5)
- **Samples per Bank**: Change dataset size (500-2000)
- **Training Epochs**: Modify training duration (5-20)

## Technical Details

### Federated Learning

The implementation uses **Federated Averaging (FedAvg)**, where:
1. Each client (bank) trains locally on private data
2. Only model weights are shared (encrypted)
3. A central server aggregates the weights
4. The global model is distributed back (optional)

### Homomorphic Encryption

Uses TenSEAL's CKKS scheme:
- **Scheme**: CKKS (approximate arithmetic on encrypted data)
- **Polynomial Modulus Degree**: 8192
- **Coefficient Modulus**: [60, 40, 40, 60] bits
- **Scale**: 2^40

This allows mathematical operations on encrypted data without decryption.

### Model Architecture

Simple binary classification neural network:
- **Input Layer**: 10 features
- **Hidden Layer**: 16 neurons with ReLU activation
- **Dropout**: 20% for regularization
- **Output Layer**: 1 neuron with sigmoid activation (binary classification)

### Data Generation

Synthetic fraud data with:
- Gaussian-distributed features
- Realistic fraud rates (5-15% per bank)
- Bank-specific patterns (slight variations in feature distributions)

## Troubleshooting

### TenSEAL Installation Issues

If TenSEAL fails to install:

```bash
# On Ubuntu/Debian
sudo apt-get install cmake build-essential

# On macOS
brew install cmake

# Then retry
pip install tenseal
```

### OpenAI API Errors

If you get API errors:
- Verify your API key is correct in `.env`
- Check your OpenAI account has credits
- Ensure the model name is valid (e.g., `gpt-3.5-turbo`)

### Memory Issues

If you encounter memory issues:
- Reduce `NUM_SAMPLES` in `.env`
- Reduce `NUM_BANKS`
- Reduce `POLY_MODULUS_DEGREE` (at the cost of security)

## Security Considerations

This is a **prototype for educational purposes**. For production use, consider:

- **Key Management**: Implement proper key distribution and management
- **Secure Communication**: Use TLS/SSL for all network communication
- **Access Control**: Implement authentication and authorization
- **Audit Logging**: Enhanced logging for compliance
- **Differential Privacy**: Add noise to protect against inference attacks
- **Secure Enclaves**: Consider using hardware security modules (HSMs)

## Performance

Typical performance on a modern CPU:

- **Data Generation**: < 1 second
- **Model Training** (per bank): 10-30 seconds
- **Encryption**: 2-5 seconds per model
- **Aggregation**: < 1 second
- **LLM Explanation**: 5-15 seconds (depends on OpenAI API)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **TenSEAL**: For homomorphic encryption capabilities
- **TensorFlow**: For machine learning framework
- **Streamlit**: For the interactive UI
- **OpenAI**: For LLM capabilities

## References

- [Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/abs/1610.05492)
- [TenSEAL: A Library for Homomorphic Encryption Operations on Tensors](https://github.com/OpenMined/TenSEAL)
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

## Support

For issues, questions, or contributions:
- **Issues**: https://github.com/almosttomorrow/federatedlearning/issues
- **Discussions**: https://github.com/almosttomorrow/federatedlearning/discussions

## Roadmap

Future enhancements:

- [ ] Add differential privacy mechanisms
- [ ] Support for more model architectures
- [ ] Real-world dataset integration
- [ ] Multi-round federated training
- [ ] Client selection strategies
- [ ] Byzantine-robust aggregation
- [ ] Model evaluation metrics
- [ ] Export to production formats

---

**Note**: This is an educational prototype demonstrating federated learning concepts. For production deployments, additional security hardening and testing are required.
