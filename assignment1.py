import statsmodels 
import numpy 
import pygam 
import ipynb
import numpy
import scipy
import plotly

import pandas as pd
from prophet import Prophet

# Load data
data = pd.read_csv("assignment_data_train.csv")

# Convert 'Timestamp' column to datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Rename columns for Prophet compatibility
data = data.rename(columns={'Timestamp': 'ds', 'trips': 'y'})

# Initialize and fit the model
model = Prophet()
modelFit = model.fit(data)

# Create a future dataframe for 744 hours (January of the next year)
future = model.make_future_dataframe(periods=744, freq='H')

# Generate predictions
forecast = model.predict(future)

# Extract predictions for the test period
pred = forecast[['ds', 'yhat']].tail(744)

# Display first few predictions
print(pred.head())

# Plot forecast
fig = model.plot(forecast)
fig.show()

print("hello")