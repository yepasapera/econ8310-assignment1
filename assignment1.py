import pandas as pd
from prophet import Prophet
import numpy as np
import scipy.stats as stats

# training data
train_data = pd.read_csv("assignment_data_train.csv")
#'Timestamp' to datetime for training data
train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'])
#ds and y recomendation
train_data = train_data.rename(columns={'Timestamp': 'ds', 'trips': 'y'})
print("hello")
# fit the model
model = Prophet()
modelFit = model.fit(train_data)
print("hello")
# test data
test_data = pd.read_csv("assignment_data_test.csv")
# 'Timestamp' to datetime test
test_data['Timestamp'] = pd.to_datetime(test_data['Timestamp'])
# comparing
test_data = test_data.rename(columns={'Timestamp': 'ds'})

#Create prediction for entire 2019
forecast = model.predict(test_data)

# numerical preditctions
pred = forecast['yhat'].to_numpy()
print("hello")
# save predictions
pd.DataFrame(pred, columns=['yhat']).to_csv("taxi_trips_forecast.csv", index=False)

# show 5 predictions
print(pred[:5])

# Plot forecast
fig = model.plot(forecast)
fig.show()
print("Hello")
# Load predictions
pred = pd.read_csv("taxi_trips_forecast.csv")['yhat'].to_numpy()

# Compare only Jan pred vs test
jan_test = test_data[test_data['ds'].dt.month == 1].copy()
jan_actual = jan_test['trips'].to_numpy()[:744]  
jan_pred = pred[:744]  

# Compute MAPE manually
mape = np.mean(np.abs((jan_actual - jan_pred) / jan_actual)) * 100

# Compute sMAPE manually
smape = np.mean(2 * np.abs(jan_pred - jan_actual) / (np.abs(jan_pred) + np.abs(jan_actual))) * 100

print(f"MAPE: {mape:.4f}%")
print(f"sMAPE: {smape:.4f}%")
print("hello")