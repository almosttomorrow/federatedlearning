import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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

# UI for synthetic data creation
st.title('Federated Learning Simulation')
if st.button('Create synthetic fraud data for three banks'):
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

# UI for federated learning aggregation
if 'models' in st.session_state and st.button('Submit for federated learning'):
    models = st.session_state['models']
    weights_list = [model.get_weights() for model in models.values()]

    # Federated aggregation
    avg_weights = [np.mean([weights[i] for weights in weights_list], axis=0) for i in range(len(weights_list[0]))]

    # Create a global model and set averaged weights
    global_model = create_model()
    global_model.set_weights(avg_weights)
    st.write("Global model weights after aggregation:")
    plot_model_weights(global_model, title="Global Model Weights")

    # Save global model weights
    with open('global_model_weights.pkl', 'wb') as f:
        pickle.dump(global_model.get_weights(), f)
    st.write("Global model weights saved.")

    # Download global model
    st.download_button('Download Global Model', 'global_model_weights.pkl')