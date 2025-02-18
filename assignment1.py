import pandas as pd
from prophet import Prophet

# training data
train_data = pd.read_csv("assignment_data_train.csv")
#'Timestamp' to datetime for training data
train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'])
#ds and y recomendation
train_data = train_data.rename(columns={'Timestamp': 'ds', 'trips': 'y'})

# fit the model
model = Prophet()
modelFit = model.fit(train_data)

# test data
test_data = pd.read_csv("assignment_data_test.csv")
# 'Timestamp' to datetime test
test_data['Timestamp'] = pd.to_datetime(test_data['Timestamp'])
# comparing
test_data = test_data.rename(columns={'Timestamp': 'ds'})

#Create prediction not for 744 but all of test data
forecast = model.predict(test_data)

# numerical preditctions
pred = forecast['yhat'].to_numpy()

# save predictions
pd.DataFrame(pred, columns=['yhat']).to_csv("taxi_trips_forecast.csv", index=False)

# show 5 predictions
print(pred[:5])

# Plot forecast
fig = model.plot(forecast)
fig.show()