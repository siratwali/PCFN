from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests
import joblib
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models
budget_model = joblib.load('bugetmodel.pkl')
next_month_model = joblib.load('nextmonthprediction.pkl')
with open('multi_output_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return "Welcome to the Combined API for Budget and Product Scraping!"

# API 1: Scrape product details from a URL
@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US, en;q=0.5'
    }
    
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the product title
        title_element = soup.find('span', id='productTitle')
        title = title_element.get_text(strip=True) if title_element else 'Title not found'
        
        # Extract the product image URL
        image_element = soup.find('div', id='imgTagWrapperId')
        image_url = image_element.find('img')['src'] if image_element else 'Image link not found'
        
        # Extract the product description
        description_element = soup.find('div', id='productDescription')
        description = description_element.get_text(strip=True) if description_element else 'Description not found'
        
        # Extract the product price
        price_element = soup.find('span', class_='a-price-whole')
        price = price_element.get_text(strip=True) if price_element else 'Price not found'
        
        return jsonify({
            "title": title,
            "image_url": image_url,
            "description": description,
            "price": price,
            "url": url
        })
    
    else:
        return jsonify({"error": f"Failed to retrieve the page. Status code: {response.status_code}"}), response.status_code

# API 2: Predict budget based on previous spending and goal
@app.route('/predict', methods=['POST'])
def predict_budget():
    data = request.json
    
    monthly_income = data.get('monthly_income')
    goal = data.get('goal')
    previous_data = data.get('previous_data', None)
    
    if previous_data:
        # User has previous spending data
        records = np.array(previous_data)
        if records.shape[1] != 10:
            return jsonify({'error': 'Each record must contain 10 values: Monthly Income and 9 expense categories.'}), 400
        
        # Use the monthly income for prediction
        monthly_incomes = records[:, 0].reshape(-1, 1)
        y_pred = budget_model.predict(monthly_incomes)
        
        # Average the predicted values
        y_pred_mean = np.mean(y_pred, axis=0)
    else:
        # User has no previous spending data
        X_new = np.array([[monthly_income]])
        y_pred_mean = budget_model.predict(X_new)[0]
        y_pred_mean = [round(expense) for expense in y_pred_mean]  # Convert to integers

    
    # Adjust the budget to achieve the goal
    total_expenses = sum(y_pred_mean)
    adjusted_budget = [(expense / total_expenses) * (monthly_income - goal) for expense in y_pred_mean]
    
    # Prepare the response
    response = {
        'suggested_budget': {
            'utilities': adjusted_budget[0],
            'food': adjusted_budget[1],
            'transportation': adjusted_budget[2],
            'healthcare': adjusted_budget[3],
            'personal_care': adjusted_budget[4],
            'entertainment': adjusted_budget[5],
            'education': adjusted_budget[6],
            'savings': adjusted_budget[7],
            'others': adjusted_budget[8],
        },
        'goal': goal
    }
    
    return jsonify(response)

# API 3: Predict expenses for the next month
@app.route('/nextmonthpredict', methods=['POST'])
def predict_next_month():
    data = request.json
    
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
    
    features_array = np.array(features).reshape(1, -1)
    prediction = next_month_model.predict(features_array)[0]
    
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

# API 4: Test Prediction
@app.route('/predicttestingg', methods=['POST'])
def predict_testing():
    data = request.json
    try:
        features = [
            data['month1_utilities'], data['month1_food'], data['month1_transportation'], 
            data['month1_healthcare'], data['month1_personal care'], data['month1_education'], 
            data['month1_entertainment'], data['month1_others'], data['month1_spending'], 
            data['month1_income'], data['month2_utilities'], data['month2_food'], 
            data['month2_transportation'], data['month2_healthcare'], data['month2_personal care'], 
            data['month2_education'], data['month2_entertainment'], data['month2_others'], 
            data['month2_spending'], data['month2_income']
        ]
    except KeyError:
        return jsonify({"error": "Invalid input data. Please provide all required fields."}), 400

    input_data = np.array([features])
    scaled_input = loaded_scaler.transform(input_data)
    prediction = loaded_model.predict(scaled_input)

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

# Run the app
if __name__ == '_main_':
    app.run(debug=True)