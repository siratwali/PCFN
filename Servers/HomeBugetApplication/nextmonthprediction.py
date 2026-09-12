from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('nextmonthprediction.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.json
    
    # Extract features for prediction
    features = [
        data['Monthly Income'],
        data['Utilities_M1'], data['Food_M1'], data['Transportation_M1'], 
        data['Healthcare_M1'], data['Personal Care_M1'], data['Entertainment_M1'], 
        data['Education_M1'], data['Savings_M1'], data['Others_M1'],
        data['Utilities_M2'], data['Food_M2'], data['Transportation_M2'], 
        data['Healthcare_M2'], data['Personal Care_M2'], data['Entertainment_M2'], 
        data['Education_M2'], data['Savings_M2'], data['Others_M2'],
        data['Utilities_M3'], data['Food_M3'], data['Transportation_M3'], 
        data['Healthcare_M3'], data['Personal Care_M3'], data['Entertainment_M3'], 
        data['Education_M3'], data['Savings_M3'], data['Others_M3']
    ]
    
    # Convert features to a numpy array and reshape for prediction
    features_array = np.array(features).reshape(1, -1)
    
    # Predict the budget for next month
    prediction = model.predict(features_array)[0]
    
    # Create a response dictionary
    response = {
        'Utilities': prediction[0],
        'Food': prediction[1],
        'Transportation': prediction[2],
        'Healthcare': prediction[3],
        'Personal Care': prediction[4],
        'Entertainment': prediction[5],
        'Education': prediction[6],
        'Savings': prediction[7],
        'Others': prediction[8]
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
