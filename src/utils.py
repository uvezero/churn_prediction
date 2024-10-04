import solara
import pandas as pd
from io import BytesIO
from solara.components.file_drop import FileInfo
import requests


# Function to load a CSV into a general DataFrame
def load_df(data: BytesIO) -> pd.DataFrame:
    new_df = pd.read_csv(data)
    return new_df



def call_api(df, single_prediction=False):
    data_json = df.to_dict(orient='list')

    # API endpoints
    predict_url = "http://127.0.0.1:5000/predict"
    explain_url = "http://127.0.0.1:5000/explain"

    try:
        # Send POST request for prediction
        response_predict = requests.post(predict_url, json=data_json)
        if response_predict.status_code != 200:
            print(f"Prediction API returned status code: {response_predict.status_code}")
            return None, None

        result = response_predict.json()
        if single_prediction:
            churn_probability = result.get("churn_probability")
            predictions = churn_probability
        else:
            churn_probabilities = result.get("churn_probabilities")
            predictions = churn_probabilities

        # Send POST request for SHAP values
        response_shap = requests.post(explain_url, json=data_json)
        if response_shap.status_code != 200:
            print(f"SHAP API returned status code: {response_shap.status_code}")
            return None, None

        shap_values = response_shap.json().get("shap_values")

        return predictions, shap_values
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None, None




def collect_encode_ui(data_dict=None, input_df=None, single_input=False):
    if input_df is None:
        # If input_df is not provided, create DataFrame from the provided dictionary.
        df = pd.DataFrame(data_dict)
        
    elif input_df is not None:
        
        df = input_df
    else:
        raise ValueError("Either input_df or reactive variables must be provided.")

    # Apply one-hot encoding to include all possible categories
    columns_to_encode = ['Contract', 'InternetService', 'PaymentMethod']
    if all(col in df.columns for col in columns_to_encode):
        df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=False)
    else:
        df_encoded = df  # If already encoded, skip one-hot encoding

    # Reorder columns
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
    
    # Reindex to maintain column order
    df_encoded = df_encoded.reindex(columns=column_order, fill_value=0)

    # Correct data types
    categorical_columns = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'PaperlessBilling'
    ]
    df_encoded[categorical_columns] = df_encoded[categorical_columns].astype('category')
    df_encoded[['SeniorCitizen', 'tenure']] = df_encoded[['SeniorCitizen', 'tenure']].astype('int64')
    df_encoded[['MonthlyCharges', 'TotalCharges']] = df_encoded[['MonthlyCharges', 'TotalCharges']].astype('float64')

    return df_encoded




@solara.component
def FileDropCSVReader(predictions_batch, shap_values_batch):
    df_state, set_df_state = solara.use_state(None)
    transformed_df_state, set_transformed_df_state = solara.use_state(None)

    def on_file(f: FileInfo):
        if not f["data"]:
            return
        new_df = load_df(BytesIO(f["data"]))
        set_df_state(new_df)

        transformed_df = collect_encode_ui(input_df=new_df, single_input=False)
        set_transformed_df_state(transformed_df)

        # Call the API to get predictions and SHAP values
        churn_probabilities, shap_vals = call_api(transformed_df)
        
        if churn_probabilities is not None:
            predictions_batch.value = churn_probabilities
            shap_values_batch.value = shap_vals
            print(f"Predictions received: {churn_probabilities}")
        else:
            print('No predictions returned')

    solara.FileDrop(
        label="Drop a CSV here",
        on_file=on_file,
        lazy=False
    )

@solara.component
def Page():
    with solara.Card(title="General CSV Reader and Predictor"):
        FileDropCSVReader()

