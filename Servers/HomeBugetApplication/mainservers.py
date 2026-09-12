from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests
import joblib
import numpy as np

app = Flask(__name__)

# Load models
budget_model = joblib.load('bugetmodel.pkl')
next_month_model = joblib.load('nextmonthprediction.pkl')

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

@app.route('/predict', methods=['POST'])
def predict_budget():
    data = request.json
    
    # Extract data from the request
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

@app.route('/nextmonthpredict', methods=['POST'])
def predict_next_month():
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
    prediction = next_month_model.predict(features_array)[0]
    
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
