import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import pickle
import requests
import tenseal as ts

# Set your OpenAI API key
openai.api_key = "Replace with your actual API key"  # Replace with your actual API key

# Function to create synthetic fraud data
def create_synthetic_data(num_banks=3):
    np.random.seed(0)
    data_dict = {}
    for bank in range(num_banks):
        num_samples = 1000
        num_features = 10
        X = np.random.randn(num_samples, num_features)
        y = np.random.randint(0, 2, size=(num_samples, 1))
        data_dict[f'Bank_{bank+1}'] = (X, y)
    return data_dict

# Function to create a Keras model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to visualize model weights
def plot_model_weights(model, title="Model Weights"):
    weights = model.get_weights()
    fig, axs = plt.subplots(1, len(weights)//2, figsize=(20, 5))
    for i, weight in enumerate(weights[0::2]):
        sns.heatmap(weight, ax=axs[i], cmap='viridis')
        axs[i].set_title(f'Layer {i+1}')
    fig.suptitle(title)
    st.pyplot(fig)

# Function to visualize synthetic data
def visualize_synthetic_data(data_dict):
    for bank, (X, y) in data_dict.items():
        st.subheader(f'{bank} - Synthetic Fraud Data')
        fig, axs = plt.subplots(2, 5, figsize=(20, 8))
        axs = axs.flatten()
        for i in range(10):
            sns.histplot(X[:, i], bins=20, ax=axs[i], color=sns.color_palette("husl", 10)[i])
            axs[i].set_title(f'Feature {i+1}')
        fig.suptitle(f'{bank} - Feature Distributions')
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(y.flatten(), bins=2, ax=ax, color='purple')
        ax.set_title('Label Distribution (0: No Fraud, 1: Fraud)')
        st.pyplot(fig)

# Initialize TenSEAL context
def initialize_tenseal():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context

context = initialize_tenseal()

# Function to encrypt weights
def encrypt_weights(weights, context):
    encrypted_weights = []
    for layer in weights:
        encrypted_layer = ts.ckks_vector(context, layer.flatten().tolist())
        encrypted_weights.append(encrypted_layer)
    return encrypted_weights

# Function to decrypt weights
def decrypt_weights(encrypted_weights):
    decrypted_weights = []
    for encrypted_layer in encrypted_weights:
        decrypted_layer = encrypted_layer.decrypt()
        decrypted_weights.append(np.array(decrypted_layer))
    return decrypted_weights

# Function to reshape decrypted weights correctly
def reshape_weights(decrypted_weights, model):
    reshaped_weights = []
    layer_shapes = [layer.shape for layer in model.get_weights()]
    start = 0
    for shape in layer_shapes:
        size = np.prod(shape)
        reshaped_weights.append(decrypted_weights[start:start+size].reshape(shape))
        start += size
    return reshaped_weights

# Function to generate explanations using OpenAI GPT
def generate_explanation(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    return response.json()['choices'][0]['message']['content'].strip()

# UI for synthetic data creation
st.title('Federated Learning Simulation')
if st.button('Create synthetic data for all banks'):
    data_dict = create_synthetic_data()
    for bank, (X, y) in data_dict.items():
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y_df = pd.DataFrame(y, columns=['label'])
        X_df.to_csv(f'{bank}_features.csv', index=False)
        y_df.to_csv(f'{bank}_labels.csv', index=False)
    st.write("Synthetic data created for all banks.")
    st.session_state['data_dict'] = data_dict
    visualize_synthetic_data(data_dict)

# UI for training models locally
if 'data_dict' in st.session_state and st.button('Train models locally'):
    data_dict = st.session_state['data_dict']
    models = {}
    for bank, (X_train, y_train) in data_dict.items():
        model = create_model()
        model.fit(X_train, y_train, epochs=10, verbose=0)
        models[bank] = model
        st.write(f"Model trained for {bank}.")
        plot_model_weights(model, title=f"{bank} Model Weights")

        # Save model weights
        with open(f'{bank}_model_weights.pkl', 'wb') as f:
            pickle.dump(model.get_weights(), f)
    st.session_state['models'] = models
    st.write("All models trained and weights saved.")

# Add a button for encryption step
if 'models' in st.session_state and st.button('Encrypt with Homomorphic Encryption'):
    encrypted_weights_dict = {}
    for bank, model in st.session_state['models'].items():
        encrypted_weights = encrypt_weights(model.get_weights(), context)
        encrypted_weights_dict[bank] = encrypted_weights
    st.session_state['encrypted_weights'] = encrypted_weights_dict
    st.write("Model weights encrypted with homomorphic encryption.")

# Visualize encryption
def visualize_encryption(weights, title="Encrypted Weights Visualization"):
    fig, ax = plt.subplots(figsize=(10, 5))
    encrypted_flattened = [len(layer.serialize()) for layer in weights]
    ax.plot(encrypted_flattened, color='blue')
    ax.set_title(title)
    st.pyplot(fig)

if 'encrypted_weights' in st.session_state:
    for bank, encrypted_weights in st.session_state['encrypted_weights'].items():
        visualize_encryption(encrypted_weights, title=f"{bank} Encrypted Weights Visualization")

# Update the federated learning aggregation step
if 'encrypted_weights' in st.session_state and st.button('Submit for federated learning'):
    encrypted_weights_list = list(st.session_state['encrypted_weights'].values())
    decrypted_weights_list = [decrypt_weights(encrypted) for encrypted in encrypted_weights_list]

    # Flatten the decrypted weights for averaging
    flat_weights_list = [np.concatenate([w.flatten() for w in weights]) for weights in decrypted_weights_list]
    avg_flat_weights = np.mean(flat_weights_list, axis=0)

    # Reshape the averaged flat weights to match the model's layer shapes
    global_model = create_model()
    layer_shapes = [layer.shape for layer in global_model.get_weights()]
    avg_weights = reshape_weights(avg_flat_weights, global_model)

    global_model.set_weights(avg_weights)
    st.write("Global model weights after aggregation and decryption:")
    plot_model_weights(global_model, title="Global Model Weights")

    with open('global_model_weights.pkl', 'wb') as f:
        pickle.dump(global_model.get_weights(), f)
    st.write("Global model weights saved.")
    st.session_state['global_model'] = global_model

    st.download_button('Download Global Model', 'global_model_weights.pkl')

# Button to generate explanation using LLM
if st.button('Generate Explanation'):
    # Prepare prompt for the LLM
    data_summary = ""
    for bank, (X, y) in st.session_state['data_dict'].items():
        data_summary += f"{bank} - Feature Means: {np.mean(X, axis=0)}, Label Distribution: {np.bincount(y.flatten())}\n"

    weights_summary = ""
    for bank, model in st.session_state['models'].items():
        weights_summary += f"{bank} - Model Weights: {model.get_weights()}\n"

    global_weights_summary = f"Global Model Weights: {st.session_state['global_model'].get_weights()}\n"

    prompt = f"""
    The following is a summary of a federated learning simulation:

    Synthetic Data is Created. Summary:
    {data_summary}

    Local Model Weights are created by training local models. Summary:
    {weights_summary}

    Local model weights are encrypted using homomorphic encryption

    Encrypted local model weights are aggregated (whilst encrypted) and then decrypted
    {global_weights_summary}

    The global model is then trained on the decrypted aggregate weights

    Please explain the federated learning process that took place, explain the type of fraud data that was created as some people will not understand what has happened here and the meaning of the data (explicitly cover how much fraud was in each bank's data set), review the global model's weights, and provide insights into how the model's performance might be affected by the federated learning approach. Please make this explanation succinct and clear (use dot points when appropriate). Open with a quick overview, then have headings for each stage in the federated process explaining exactly what is happening and why. The idea of this is to educate and explain what is happening with the unique data that has been created in this example. Finish with a simple conclusion on what happened and why federated learning is used in this context.
    """
    explanation = generate_explanation(prompt)
    st.write(explanation)
