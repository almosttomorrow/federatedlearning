# Federated Learning Simulation

This project demonstrates a federated learning simulation using TensorFlow and Streamlit. The application simulates three banks creating synthetic fraud data, training local models, and aggregating model weights to form a global model.

## Features

- Create synthetic fraud data for three banks
- Train local models using TensorFlow
- Visualize model weights
- Aggregate model weights to form a global model
- Download the global model

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/almosttomorrow/federatedlearning.git
    cd federatedlearning
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Follow the UI to create synthetic data, train models, and aggregate model weights.

## Files

- `app.py`: Main application script.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
