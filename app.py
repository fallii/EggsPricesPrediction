import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['Tahun'])
    month = float(request.form['Bulan'])

    # Use the model to predict egg prices
    predicted_price = model.predict([[year, month]])

    return render_template('index.html', prediction_text="{}".format(predicted_price))


if __name__ == '__main__':
    app.run(debug=True)

