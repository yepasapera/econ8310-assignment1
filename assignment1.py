import statsmodels 
import numpy 
import pygam 
import ipynb
import numpy
import scipy
import plotly

import pandas as pd
from prophet import Prophet

# Load training data
train_data = pd.read_csv("assignment_data_train.csv")

# Convert 'Timestamp' column to datetime format
train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'])

# Rename columns for Prophet compatibility
train_data = train_data.rename(columns={'Timestamp': 'ds', 'trips': 'y'})

# Initialize and fit the model
model = Prophet()
modelFit = model.fit(train_data)

# Load test data
test_data = pd.read_csv("assignment_data_test.csv")

# Convert 'Timestamp' column to datetime format
test_data['Timestamp'] = pd.to_datetime(test_data['Timestamp'])

# Rename columns for Prophet
test_data = test_data.rename(columns={'Timestamp': 'ds'})

# Generate predictions for the test period
forecast = model.predict(test_data)

# Extract predictions
pred = forecast[['ds', 'yhat']]

# Save predictions to CSV
pred.to_csv("taxi_trips_forecast.csv", index=False)

# Display first few predictions
print(pred.head())

# Plot forecast
fig = model.plot(forecast)
fig.show()

