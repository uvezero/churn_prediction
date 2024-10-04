# Use a Python-based image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and the model
COPY ./src /app/src
COPY ./models/catboost_model.cbm /app/models/catboost_model.cbm  

# Install Supervisor to run multiple services
RUN apt-get update && apt-get install -y supervisor

# Copy the Supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose necessary ports for Flask API and Solara Interface
EXPOSE 5000 8765

# Run Supervisor to manage both Flask and Solara
CMD ["/usr/bin/supervisord"]
