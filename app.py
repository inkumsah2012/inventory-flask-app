from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model directly from root folder
model = tf.keras.models.load_model('inventory_model.keras')

# Load the scaler from root folder
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define number of time steps used in LSTM input
time_steps = 30

# Load the recent historical data
data_path = 'inventory_data_large.csv'
historical_data = pd.read_csv(data_path)
recent_history = historical_data['Consumption'].values[-(time_steps - 1):]

@app.route('/')
def index():
    return render_template('index.html')

def ValuePredictor(consumption):
    # Append today's consumption to the recent history
    consumption_sequence = np.append(recent_history, consumption).reshape(-1, 1)

    # Add placeholder second column (Ending_Quantity = 0)
    placeholder_column = np.zeros((consumption_sequence.shape[0], 1))
    combined_sequence = np.hstack((consumption_sequence, placeholder_column))

    # Scale using the fitted scaler
    scaled_data = scaler.transform(combined_sequence)

    # Reshape for LSTM: (1 sample, 30 time steps, 1 feature)
    X_input = scaled_data[:, 0].reshape(1, time_steps, 1)

    # Make prediction and inverse transform the scaled output
    prediction_scaled = model.predict(X_input)
    prediction_rescaled = scaler.inverse_transform([[0, prediction_scaled[0, 0]]])[0][1]

    return prediction_rescaled

@app.route('/predict', methods=['POST'])
def result():
    try:
        # Get and validate input
        consumption_input = request.form.get('consumption', '')
        consumption = float(consumption_input)

        # Make prediction
        prediction = ValuePredictor(consumption)
        return render_template('predict.html', prediction=round(prediction, 2))

    except ValueError:
        return jsonify({"error": "Invalid input: Please enter a numeric consumption value."})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=False)
