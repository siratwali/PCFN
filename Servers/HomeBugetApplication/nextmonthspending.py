from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model and scaler from pickle files
with open('multi_output_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return "Welcome to the Budget Prediction API!"

# Define an API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the POST request (JSON format)
    data = request.json
    
    # Ensure the correct number of features are provided
    try:
        features = [data['month1_utilities'], data['month1_food'], data['month1_transportation'], data['month1_healthcare'], 
                    data['month1_personal care'], data['month1_education'], data['month1_entertainment'], data['month1_others'], 
                    data['month1_spending'], data['month1_income'], data['month2_utilities'], data['month2_food'], 
                    data['month2_transportation'], data['month2_healthcare'], data['month2_personal care'], data['month2_education'], 
                    data['month2_entertainment'], data['month2_others'], data['month2_spending'], data['month2_income']]
    except KeyError:
        return jsonify({"error": "Invalid input data. Please provide all required fields."}), 400
    
    # Convert the input data to a NumPy array and reshape it for scaling
    input_data = np.array([features])
    
    # Scale the input data using the loaded scaler
    scaled_input = loaded_scaler.transform(input_data)
    
    # Make predictions using the loaded model
    prediction = loaded_model.predict(scaled_input)
    
    # Return the prediction as JSON
    return jsonify({
        'month3_utilities': prediction[0][0],
        'month3_food': prediction[0][1],
        'month3_transportation': prediction[0][2],
        'month3_healthcare': prediction[0][3],
        'month3_personal_care': prediction[0][4],
        'month3_education': prediction[0][5],
        'month3_entertainment': prediction[0][6],
        'month3_others': prediction[0][7],
        'month3_spending': prediction[0][8],
        'month3_income': prediction[0][9]
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
