import pickle
import numpy as np
import pandas as pd  # Import Pandas
from flask import Flask, request, render_template

# Load the saved model
model = pickle.load(open('saved_model.sav', 'rb'))

# Initialize the Flask application
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input values safely
        sepal_length = request.form.get('sepal_length', '').strip()
        sepal_width = request.form.get('sepal_width', '').strip()
        petal_length = request.form.get('petal_length', '').strip()
        petal_width = request.form.get('petal_width', '').strip()

        # Ensure all fields have values
        if not sepal_length or not sepal_width or not petal_length or not petal_width:
            return render_template('index.html', result="Error: All fields are required!")

        # Convert inputs to float
        sepal_length = float(sepal_length)
        sepal_width = float(sepal_width)
        petal_length = float(petal_length)
        petal_width = float(petal_width)

        # Convert input data to a Pandas DataFrame (ensuring it has column names)
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        input_features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_names)

        # Debugging print statements
        print(f"Received Input Data:\n{input_features}")

        # Make prediction
        result = model.predict(input_features)[0]

        # Print prediction for debugging
        print(f"Prediction Result: {result}")

        return render_template('index.html', result=result)

    except ValueError:
        return render_template('index.html', result="Error: Please enter valid numeric values!")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
