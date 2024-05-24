
import streamlit as st
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def create_keras_model():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(10,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(tf.TensorSpec(shape=[None, 10], dtype=tf.float32),
                    tf.TensorSpec(shape=[None], dtype=tf.float32)),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

iterative_process = tff.learning.algorithms.build_weighted_fed_avg(model_fn)
state = iterative_process.initialize()

def create_fake_data():
    data = []
    for _ in range(3):
        x = np.random.normal(size=(100, 10))
        y = np.random.randint(2, size=(100, 1))
        df = pd.DataFrame(np.hstack((x, y)), columns=[f'feature_{i}' for i in range(10)] + ['label'])
        data.append(df)
    return data

data = create_fake_data()

st.title('Federated Learning with TFF Demonstration')

if st.button('Initialize State'):
    state = iterative_process.initialize()
    st.write("State initialized.")

uploaded_files = []
for i in range(3):
    uploaded_file = st.file_uploader(f"Upload data for Bank {i+1}", type="csv", key=i)
    if uploaded_file:
        uploaded_files.append(uploaded_file)

if st.button('Submit Data and Train Locally'):
    clients_data = []
    for file in uploaded_files:
        df = pd.read_csv(file)
        x_train = df.iloc[:, :-1].values
        y_train = df.iloc[:, -1].values
        clients_data.append(tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(10))
    st.write("Training locally...")
    for round_num in range(1, 11):
        state, metrics = iterative_process.next(state, clients_data)
        st.write(f'Round {round_num}, Metrics={metrics}')

if st.button('Aggregate and Download Global Model'):
    st.write("Aggregating and updating global model...")
    for round_num in range(1, 11):
        state, metrics = iterative_process.next(state, clients_data)
    st.write("Aggregation complete. Download the latest model.")
    with open('global_model.pkl', 'wb') as f:
        pickle.dump(state.model, f)
    st.download_button('Download Global Model', 'global_model.pkl')

def plot_model_weights(model, title="Model Weights"):
    weights = model.get_weights()
    fig, axs = plt.subplots(1, len(weights), figsize=(20, 5))
    for i, weight in enumerate(weights):
        axs[i].imshow(weight, aspect='auto', cmap='viridis')
        axs[i].set_title(f'Layer {i+1}')
    st.pyplot(fig)

if st.button('Visualize Local Model Weights'):
    for i, file in enumerate(uploaded_files):
        df = pd.read_csv(file)
        x_train = df.iloc[:, :-1].values
        y_train = df.iloc[:, -1].values
        local_model = create_keras_model()
        local_model.compile(optimizer='adam', loss='binary_crossentropy')
        local_model.fit(x_train, y_train, epochs=1, verbose=0)
        st.write(f'Bank {i+1} Model Weights')
        plot_model_weights(local_model)

if st.button('Visualize Global Model Weights'):
    global_model = create_keras_model()
    global_model.set_weights(state.model.weights)
    st.write("Global Model Weights")
    plot_model_weights(global_model)
    """)
