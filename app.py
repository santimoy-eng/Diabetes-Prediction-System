# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'classifier.pkl'
scaler_path='scaler.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    input_data = [float(x) for x in request.form.values()]
    # input_data = [1,85,66,29,0,26.6,0.351,31]

    input_data=np.array([input_data])

    std_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(std_data)
    # output = 'Placed' if prediction[0] == 1 else 'Not Placed'
    if(prediction[0]==0):
        # print('The person is non-Diabetic')
        output="Non Diabetic"
    else:
        # print('The person is Diabetic')
        output="Diabetic"

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)