import pandas as pd
from catboost import CatBoostClassifier
import os

MODEL_PATH = "/app/models/catboost_model.cbm"  # Path inside Docker
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

# Encode user input
def collect_encode_ui(user_input):
    data = {
        'gender': [user_input['gender']],
        'SeniorCitizen': [user_input['SeniorCitizen']],
        'Partner': [user_input['Partner']],
        'Dependents': [user_input['Dependents']],
        'tenure': [user_input['tenure']],
        'PhoneService': [user_input['PhoneService']],
        'MultipleLines': [user_input['MultipleLines']],
        'OnlineSecurity': [user_input['OnlineSecurity']],
        'OnlineBackup': [user_input['OnlineBackup']],
        'DeviceProtection': [user_input['DeviceProtection']],
        'TechSupport': [user_input['TechSupport']],
        'StreamingTV': [user_input['StreamingTV']],
        'StreamingMovies': [user_input['StreamingMovies']],
        'PaperlessBilling': [user_input['PaperlessBilling']],
        'MonthlyCharges': [user_input['MonthlyCharges']],
        'TotalCharges': [user_input['TotalCharges']],
        'Contract': [user_input['Contract']],
        'InternetService': [user_input['InternetService']],
        'PaymentMethod': [user_input['PaymentMethod']]
    }

    df = pd.DataFrame(data)

    columns_to_encode = ['Contract', 'InternetService', 'PaymentMethod']
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=False)

    column_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]

    df_encoded = df_encoded.reindex(columns=column_order, fill_value=0)

    return df_encoded


def predict_churn(user_input):
    try:
        encoded_data = collect_encode_ui(user_input)
        prediction = model.predict_proba(encoded_data)[:, 1][0]
        return {"Churn Probability": float(prediction)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    customer_input = {
        "gender": input("Gender (Male/Female): ").strip(),
        "SeniorCitizen": int(input("Senior Citizen (0/1): ").strip()),
        "Partner": input("Partner (Yes/No): ").strip(),
        "Dependents": input("Dependents (Yes/No): ").strip(),
        "tenure": int(input("Tenure (months): ").strip()),
        "PhoneService": input("Phone Service (Yes/No): ").strip(),
        "MultipleLines": input("Multiple Lines (Yes/No): ").strip(),
        "InternetService": input("Internet Service (DSL/Fiber optic/No): ").strip(),
        "OnlineSecurity": input("Online Security (Yes/No): ").strip(),
        "OnlineBackup": input("Online Backup (Yes/No): ").strip(),
        "DeviceProtection": input("Device Protection (Yes/No): ").strip(),
        "TechSupport": input("Tech Support (Yes/No): ").strip(),
        "StreamingTV": input("Streaming TV (Yes/No): ").strip(),
        "StreamingMovies": input("Streaming Movies (Yes/No): ").strip(),
        "Contract": input("Contract (Month-to-month/One year/Two year): ").strip(),
        "PaperlessBilling": input("Paperless Billing (Yes/No): ").strip(),
        "PaymentMethod": input("Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)): ").strip(),
        "MonthlyCharges": float(input("Monthly Charges ($): ").strip()),
        "TotalCharges": float(input("Total Charges ($): ").strip())
    }

    result = predict_churn(customer_input)

    if "Churn Probability" in result:
        formatted_churn_probability = "{:.2%}".format(result["Churn Probability"])
        print(f"Churn Probability: {formatted_churn_probability}")
    else:
        print(f"Error: {result['error']}")

