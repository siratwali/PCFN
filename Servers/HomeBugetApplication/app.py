from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('bugetmodel.pkl')

@app.route('/predict', methods=['POST'])
def predict():
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
        y_pred = model.predict(monthly_incomes)
        
        # Average the predicted values
        y_pred_mean = np.mean(y_pred, axis=0)
    else:
        # User has no previous spending data
        X_new = np.array([[monthly_income]])
        y_pred_mean = model.predict(X_new)[0]
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

if __name__ == '__main__':
    app.run(debug=True)
