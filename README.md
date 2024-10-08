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

## Running the Project Locally

Follow these steps to run the Churn Prediction project on your local machine.

### Prerequisites

Make sure you have the following installed:
- **Git**: [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)

### Steps to Run the Project

1. **Clone the GitHub Repository**:
   First, clone the repository to your local machine using Git:

   ```bash
   git clone https://github.com/uvezero/churn_prediction.git
   cd churn_prediction

2. **Build the Docker Image:**
   Once inside the project directory, build the Docker image using the provided `dockerfile`:

    ```bash
   docker build -t churn_prediction_app .
    ```
3. **Run the Docker Container:**

  ```bash
  docker run -p 5000:5000 -p 8765:8765 churn_prediction_app
  ```
4. **Open the solara interface in you browser:**
     ```
     http://localhost:8765
