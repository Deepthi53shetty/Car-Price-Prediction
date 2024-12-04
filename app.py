# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from form
    data = request.form
    name = data['name']
    company = data['company']
    year = int(data['year'])
    kms_driven = int(data['kms_driven'])
    fuel_type = data['fuel_type']
    
    # Prepare the input for the model
    input_df = pd.DataFrame([[name, company, year, kms_driven, fuel_type]], 
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    
    # Predict the price
    prediction = model.predict(input_df)[0]
    
    return render_template('index.html', prediction_text=f'Estimated Price: ₹{int(prediction):,}')

if __name__ == '__main__':
    app.run(debug=True)
