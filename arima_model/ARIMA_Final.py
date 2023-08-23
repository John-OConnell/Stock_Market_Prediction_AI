# %%
import warnings
warnings.filterwarnings('ignore')
#Data Manipulation and Treatment
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt

#Plotting and Visualizations
import matplotlib.pyplot as plt

#Statistics
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from pmdarima import auto_arima

# %%
#Download stock data
df = yf.download('aapl', start='2010-01-01', end='2020-01-01')
df = df.resample('D').asfreq()
df = df.ffill()
df.head()

# %%
#Plot closing price of stock data
plt.figure(figsize=(10,6))
plt.grid(True)
plt.plot(df["Close"], label='Original')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close', fontsize=12)
plt.title('Apple Closing Price')

# %%
# Checking Trend and Seasonality
plt.figure(figsize=(15,7))
plt.grid(True)
plt.plot(df["Close"], label='Original')
plt.plot(df["Close"].rolling(window=12).mean(), color='red', label='Rolling mean')
plt.plot(df["Close"].rolling(window=12).std(), color='green', label='Rolling std')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close', fontsize=12)
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')

# %%
# Checking Trend and Seasonality
from statsmodels.tsa.seasonal import seasonal_decompose 
decomposition = seasonal_decompose(df['Close'], model='multiplicative', period=30)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
    
fig = decomposition.plot()
fig.set_size_inches(14, 7)

# %%
#Running ADF Test on raw data
def ADF_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    if dftest[1] > 0.05:
        print('Cannot reject the Null Hypothesis: Data not stationary.')
    else:
        print('Reject the Null Hypothesis: Data is stationary.')
ADF_test(df['Close'])

# %%
#Split data into train and training set
df_train, df_test = df[:int(len(df)*0.8)], df[int(len(df)*0.8):]

#Plot split data
y_pred=df_test.copy()
plt.figure(figsize=(10,6))
plt.grid(True)
plt.plot(df_train['Close'], label='Train data')
plt.plot(df_test['Close'], 'red', label='Test data')
plt.xlabel('Dates')
plt.ylabel('Close')
plt.legend()
plt.title('Apple Closing Price')

# %%
#Apply first order differencing to data
df_train_diff = df_train['Close'].diff().dropna()

#Plot differenced data
plt.figure(figsize=(15,7))
plt.grid(True)
plt.plot(df_train_diff, label='Original')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close', fontsize=12)
plt.title('First Order Differencing')

# %%
#Perform ADF test on differenced data
ADF_test(df_train_diff)

# %%
#Plot PACF
plt.rc("figure", figsize=(10,6))
pacf = plot_pacf(df_train_diff)

# %%
#Plot ACF
acf = plot_acf(df_train_diff)

# %%
#Fit ARIMA model to data
model_arima= auto_arima(df_train['Close'],trace=True, 
                        error_action='ignore', 
                        start_p=1,
                        start_q=1,
                        max_p=5,
                        max_q=5,
                        suppress_warnings=True,
                        stepwise=False,
                        seasonal=False)
model_arima.fit(df_train['Close'])


# %%
#Use model to make predictions on test data
prediction_arima=model_arima.predict(len(df_test))
y_pred["ARIMA Model Prediction"]=prediction_arima.dropna()

#Plot results
plt.figure(figsize=(15,7))
plt.plot(df_train["Close"], label='Train Data')
plt.plot(df_test["Close"], color='red', label='Test Data')
plt.plot(y_pred["ARIMA Model Prediction"], color='green', label='Prediction')
plt.grid(True)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close', fontsize=12)
plt.legend(loc='best')
plt.title('ARIMA(2,1,2) Model Results')

# %%
#Plot Results
plt.figure(figsize=(15,7))
plt.plot(df_test["Close"], color='red', label='Test Data')
plt.plot(y_pred["ARIMA Model Prediction"], color='green', label='Prediction')
plt.grid(True)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close', fontsize=12)
plt.legend(loc='best')
plt.title('ARIMA(2,1,2) Model Results')

# %%
#Calculate Mean Squared Error and Mean Absolute Percentage Error
mse_arima= mean_squared_error(y_pred["Close"],y_pred["ARIMA Model Prediction"])
mape_arima=mean_absolute_percentage_error(y_pred["Close"],y_pred["ARIMA Model Prediction"])
print("Mean Square Error ARIMA: ",mse_arima)
print("Mean Absoulute Percentage Error ARIMA: ",mape_arima)

# %%
df_train_new = df_train["Close"]

# %%
history = [x for x in df_train["Close"]]
test_data = [x for x in df_test["Close"]]
model_predictions = []
N_test_observations = len(df_test)

# %%
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)

for time_point in range(N_test_observations):
    model = ARIMA(history, order=(2,1,2))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)

# %%
# update ARIMA Model Predictions
y_pred["ARIMA Model Prediction"]=model_predictions

#Plot results
plt.figure(figsize=(15,7))
plt.plot(df_train["Close"], label='Train Data')
plt.plot(df_test["Close"], color='red', label='Test Data')
plt.plot(y_pred["ARIMA Model Prediction"], color='green', label='Prediction')
plt.grid(True)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close', fontsize=12)
plt.legend(loc='best')
plt.title('ARIMA(2,1,2) Model Results')

# %%
mse_arima= mean_squared_error(y_pred["Close"],y_pred["ARIMA Model Prediction"])
mape_arima=mean_absolute_percentage_error(y_pred["Close"],y_pred["ARIMA Model Prediction"])
print("Mean Square Error ARIMA: ",mse_arima)
print("Mean Absoulute Percentage Error ARIMA: ",mape_arima)

# %%
#Plot Results
plt.figure(figsize=(15,7))
plt.plot(df_test["Close"], color='red', label='Test Data')
plt.plot(y_pred["ARIMA Model Prediction"], color='green', label='Prediction')
plt.grid(True)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close', fontsize=12)
plt.legend(loc='best')
plt.title('ARIMA(2,1,2) Model Results')


