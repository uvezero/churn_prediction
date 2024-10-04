from flask import Flask, request, jsonify
import subprocess
from catboost import CatBoostClassifier
import shap
import pandas as pd

app = Flask(__name__)

#Change this if run locally to # "/home/.../catboost_model.cbm"
MODEL_PATH = "/app/models/catboost_model.cbm"  
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

# SHAP explainer
explainer = shap.TreeExplainer(model)


@app.route('/')
def home():
    return "Welcome to the Churn Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame.from_dict(data)

        churn_probabilities = model.predict_proba(df)[:, 1]

        # Predictions for batch or single case
        if len(churn_probabilities) == 1:
            churn_probability = float(churn_probabilities[0])
            return jsonify({'churn_probability': churn_probability}), 200
        else:
            return jsonify({'churn_probabilities': churn_probabilities.tolist()}), 200
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 500

    
@app.route('/explain', methods=['POST'])
def explain():
    try:
        # Parse the incoming JSON data
        data = request.json  # Expecting customer data in JSON format
        df = pd.DataFrame.from_dict(data)
        
        shap_values = explainer.shap_values(df) 
        
        return jsonify({
            'shap_values': shap_values.tolist()
        }), 200
    except Exception as e:
        print(f"Error in /explain: {e}")
        return jsonify({'error': str(e)}), 500
    
# If running locally (e.g., during development), use this:
# if __name__ == '__main__':
#     app.run(debug=True)  # Runs on localhost with debugging enabled

# For production (e.g., in Docker), use the following to expose the app externally:
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


    