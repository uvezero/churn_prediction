import solara
import pandas as pd
from catboost import CatBoostClassifier
import time
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64
import requests
import numpy as np
from utils import FileDropCSVReader
from utils import call_api
from utils import collect_encode_ui
from io import StringIO



# TO DO: Improve function access to dataframe columns
fts = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]

# Reactive variables
#Single customer inputs
gender = solara.reactive("Male")
partner = solara.reactive("Yes")
dependents = solara.reactive("Yes")
tenure = solara.reactive(72)
phone_service = solara.reactive("Yes")
multiple_lines = solara.reactive("Yes")
online_security = solara.reactive("Yes")
online_backup = solara.reactive("Yes")
device_protection = solara.reactive("Yes")
tech_support = solara.reactive("No")
streaming_tv = solara.reactive("Yes")
streaming_movies = solara.reactive("No")
paperless_billing = solara.reactive("Yes")
monthly_charges = solara.reactive(50.75)
total_charges = solara.reactive(610.00)
contract = solara.reactive("Two year")
internet_service = solara.reactive("Fiber optic")
payment_method = solara.reactive("Credit card (automatic)")

loading = solara.reactive(False)
prediction_message = solara.reactive("")
file_upload_message = solara.reactive("")
probability_value = solara.reactive(0) 
churn_gauge = solara.reactive(None)  
shap_values = solara.reactive(None)
shap_values_csv = solara.reactive(None)
show_info = solara.reactive(False)
uploaded_file = solara.reactive(None)
show_shap_batch_plot = solara.reactive(False)
predictions_batch = solara.reactive(None)
shap_values_batch = solara.reactive(None)
transformed_df_state = solara.reactive(None)

# Global variables
df = None 
transformed_df = None


# Shap Bar plot for entire data predictions
def show_shap_batch(feats, shap_values_batch):
    if shap_values_batch.value is not None:
        shap_values_array = np.array(shap_values_batch.value)

        if shap_values_array.shape[1] != len(feats):
            print(f"Error: SHAP values and feature list have different column counts.")
            return

        # Generate SHAP bar plot for the first row
        shap_explanation = shap.Explanation(values=shap_values_array, feature_names=feats)

        plt.figure(figsize=(12, 12))
        shap.plots.bar(shap_explanation[0], show=False, max_display=15)  # First row SHAP
        plt.xlabel("Contribution")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.clf()

        img_bar = Image.open(buf)
        solara.Markdown("## Feature Importance for Batch Prediction")
        solara.Image(img_bar)
    else:
        print("SHAP values are not available.")




def make_prediction():
    global df
    loading.value = True
    show_info.set(False) 
    try:
        data_dict = {
            'gender': [gender.value],
            'SeniorCitizen': [0],
            'Partner': [partner.value],
            'Dependents': [dependents.value],
            'tenure': [tenure.value],
            'PhoneService': [phone_service.value],
            'MultipleLines': [multiple_lines.value],
            'OnlineSecurity': [online_security.value],
            'OnlineBackup': [online_backup.value],
            'DeviceProtection': [device_protection.value],
            'TechSupport': [tech_support.value],
            'StreamingTV': [streaming_tv.value],
            'StreamingMovies': [streaming_movies.value],
            'PaperlessBilling': [paperless_billing.value],
            'MonthlyCharges': [monthly_charges.value],
            'TotalCharges': [total_charges.value],
            'Contract': [contract.value],
            'InternetService': [internet_service.value],
            'PaymentMethod': [payment_method.value]
        }
        
        df = collect_encode_ui(data_dict=data_dict)  
        print(f"Collected data: {df}")
        
        # Call the API
        churn_probability, shap_values_result = call_api(df,single_prediction=True) 
          
            
        if churn_probability is not None:
            probability_value.value = churn_probability * 100
            shap_values.value = np.array(shap_values_result)
            
            # Update gauge chart
            churn_gauge.value = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability_value.value,
                title={'text': "Churn Probability"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "red" if churn_probability >= 0.5 else "green"}}
            ))

            prediction_message.value = "The customer will churn" if churn_probability >= 0.5 else "The customer will not churn"
        else:
            prediction_message.value = "Error: Could not retrieve prediction"
            print("Error: Churn probability returned None")
    except Exception as e:
        print(f"An error occurred: {e}")
        prediction_message.value = "An error occurred during prediction"
    finally:
        loading.value = False



def show_shap():
    global df
    if df is None:
        print("Error: Dataframe (df) is None. Ensure it is correctly assigned.")
        return

    if shap_values.value is not None:
        # SHAP bar plot
        shap_explanation = shap.Explanation(values=shap_values.value, feature_names=df.columns)

        plt.figure(figsize=(12, 12)) 
        shap.plots.bar(shap_explanation[0], show=False, max_display=15)  # Bar plot for the first observation
        plt.xlabel("Contribution")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')  # Use bbox_inches to prevent cutting
        buf.seek(0)
        plt.clf()  

        img_bar = Image.open(buf)
        solara.Markdown("## Feature Importance for a Single Customer's Churn Prediction")
        solara.Markdown("This bar chart shows how different customer features (e.g., gender, contract type) contribute to the prediction of whether the customer will churn or not. Features on the right (shown in magenta) contribute positively to the model's prediction—either increasing the likelihood of the customer churning (if the prediction is 'churn') or decreasing it (if the prediction is 'no churn'). Features on the left (shown in blue) contribute negatively—either reducing the likelihood of churn or increasing it, depending on the prediction.")
        solara.Image(img_bar)
    else:
        print("SHAP values are not available.")

def get_predictions_csv():
    if predictions_batch.value is not None:
        # DataFrame from predictions
        df = pd.DataFrame(predictions_batch.value, columns=["Churn Prediction"])
        output = StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()  # Return the CSV data as a string
    return "No predictions available."


@solara.component
def Page():
    solara.Title("Customer Churn Interface by uvezero")

    solara.Markdown("# Customer Churn Prediction")

    with solara.Sidebar():
        solara.Markdown("### Enter Customer Information:")

        with solara.Card(title="Customer Details"):
            solara.Select(label="Gender", value=gender, values=["Female", "Male"])
            solara.Select(label="Partner", value=partner, values=["Yes", "No"])
            solara.Select(label="Dependents", value=dependents, values=["Yes", "No"])
            solara.InputFloat("Tenure (months)", value=tenure)
            solara.Select(label="Contract", value=contract, values=["Month-to-month", "One year", "Two year"])
            solara.Select(label="Payment Method", value=payment_method, values=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

        with solara.Card(title="Services"):
            solara.Select(label="Phone Service", value=phone_service, values=["Yes", "No"])
            solara.Select(label="Multiple Lines", value=multiple_lines, values=["Yes", "No"])
            solara.Select(label="Internet Service", value=internet_service, values=["DSL", "Fiber optic", "No"])
            solara.Select(label="Streaming TV", value=streaming_tv, values=["Yes", "No"])
            solara.Select(label="Streaming Movies", value=streaming_movies, values=["Yes", "No"])

        with solara.Card(title="Financial Information"):
            solara.InputFloat("Monthly Charges", value=monthly_charges)
            solara.InputFloat("Total Charges", value=total_charges)

        with solara.Card(title='Others'):
            solara.Select(label="Online Security", value=online_security, values=["Yes", "No"])
            solara.Select(label="Online Backup", value=online_backup, values=["Yes", "No"])
            solara.Select(label="Device Protection", value=device_protection, values=["Yes", "No"])
            solara.Select(label="Tech Support", value=tech_support, values=["Yes", "No"])
            solara.Select(label="Paperless Billing", value=paperless_billing, values=["Yes", "No"])
    
    
    
    solara.Markdown("### Single Prediction")        
    solara.Button(label="Submit single data", on_click=make_prediction)
    if loading.value:
        solara.Info("Predicting...", icon="spinner")
    
    if prediction_message.value:
        solara.Markdown(f"### {prediction_message.value}")

        if churn_gauge.value:
            solara.FigurePlotly(churn_gauge.value)


    solara.Button(label="Analyze the prediction", on_click=lambda: show_info.set(True)) # Make sure SHAP plot appears when needed
    
    if show_info.value and shap_values.value is not None:
        show_shap()
    
    # Render the CSV input
    solara.Markdown("### Upload CSV for Batch Predictions")
    FileDropCSVReader(predictions_batch, shap_values_batch)  # Ensure this is rendering here

    solara.Button("Show SHAP Plot for Batch", on_click=lambda: show_shap_batch_plot.set(True))
    if predictions_batch.value is not None:
        solara.FileDownload(data=get_predictions_csv, filename="predictions_batch.csv", label="Download Predictions")

    if shap_values_batch.value is not None:
        show_shap_batch(fts, shap_values_batch)

    else:
        solara.Markdown("### No batch predictions available yet")
    
