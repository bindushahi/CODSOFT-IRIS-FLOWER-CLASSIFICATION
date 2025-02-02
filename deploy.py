import pickle
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
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Retrieve input values from HTML form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])  # Fixed the typo

    # Prepare the input data
    input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
    result = model.predict(input_features)[0]

    # Render the result on the webpage
    return render_template('index.html', prediction=result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
