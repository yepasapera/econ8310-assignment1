import statsmodels 
import numpy 
import pygam 
import ipynb
import numpy
import scipy
import plotly

import pandas as pd
from prophet import Prophet

# Load the training data
train_data = pd.read_csv("assignment_data_train.csv")

# Ensure correct datetime format
train_data["Timestamp"] = pd.to_datetime(train_data["Timestamp"])

# Rename columns for Prophet
train_data = train_data.rename(columns={"Timestamp": "ds", "trips": "y"})

# Ensure no NaN values
train_data = train_data.dropna()

# Initialize Prophet model
model = Prophet()
model.add_seasonality(name="weekly", period=7, fourier_order=3)
model.add_seasonality(name="daily", period=1, fourier_order=3)

# Fit the model
model.fit(train_data)

# Load the test dataset
test_data = pd.read_csv("assignment_data_test.csv")
test_data["Timestamp"] = pd.to_datetime(test_data["Timestamp"])

# Create a dataframe for future predictions using test timestamps
future = pd.DataFrame({"ds": test_data["Timestamp"]})

# Generate forecasts
forecast = model.predict(future)

# Select only required columns
pred = forecast[["ds", "yhat"]]
pred = pred.rename(columns={"ds": "Timestamp", "yhat": "trips"})

# Save predictions to CSV
pred.to_csv("taxi_trips_forecast.csv", index=False)

print("✅ Predictions saved to taxi_trips_forecast.csv")
