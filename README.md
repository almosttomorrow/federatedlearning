# Federated Learning Prototype with Homomorphic Encryption and LLM Explanations

## Introduction
This repository contains a prototype implementation of federated learning using homomorphic encryption and an LLM to generate explanations. The prototype demonstrates how multiple entities can collaboratively train a machine learning model without sharing raw data, ensuring privacy and security.

## Key Features
- **Federated Learning**: Train models locally and aggregate weights on a central server.
- **Homomorphic Encryption**: Securely encrypt model weights during aggregation.
- **LLM Explanations**: Generate natural language explanations using OpenAI's API.

## Technologies Used
- **Google Colab**: Cloud-based Jupyter notebook environment.
- **TensorFlow**: Machine learning framework.
- **TenSEAL**: Library for homomorphic encryption on tensors.
- **Streamlit**: Framework for interactive web applications.
- **OpenAI**: API for generating natural language explanations.

## Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/almosttomorrow/federatedlearning.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Usage
1. Create synthetic fraud data for multiple banks.
2. Train local models on the synthetic data.
3. Encrypt the model weights using homomorphic encryption.
4. Aggregate encrypted weights and update the global model.
5. Generate explanations using an LLM.

## File Structure
- `app.py`: Main application code.
- `requirements.txt`: List of required Python packages.
- `.gitignore`: Files and directories to be ignored by Git.
- `README.md`: This readme file.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
