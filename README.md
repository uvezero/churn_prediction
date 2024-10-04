# Customer Churn Prediction

This project is a machine learning application designed to predict customer churn using a pre-trained CatBoost model. The project includes a Flask API for making predictions and a Solara interface for interacting with the model.

## Project Structure

- **src/**: Contains the source code for both the Flask API and Solara interface.
- **notebooks/**: Jupyter notebooks used for exploratory data analysis (EDA) and model experimentation.
- **data/**: Final dataset used for training the model.
- **models/**: Contains the pre-trained CatBoost model (`catboost_model.cbm`).
- **requirements.txt**: Lists the required Python dependencies for the project.
- **dockerfile**: Used to build the Docker image and run both Flask and Solara.
- **supervisord.conf**: Configuration file for Supervisor to manage Flask and Solara processes.

## Features

- **Flask API**: Provides an endpoint to make predictions based on customer data.
- **Solara Interface**: A web-based interface for interacting with the model.
- **Machine Learning**: The model is a pre-trained CatBoost classifier.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
