
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
file = "sp500_data.csv"

# reading single file as variable through the above attribute
train = pd.read_csv(f'data/{file}')

# Manipulation based on the Date and closing data

train = train[['Date', 'Close']]
# print(train.head().to_string())
# print(train.to_string())

# formating data to ensure that the prices are floats and the date are in the proper format
train = train.replace({'\$':''}, regex= True)
train = train.astype({"Close": float})
train["Date"] = pd.to_datetime(train["Date"], format="%Y/%m/%d")
# check data type
# print(train.dtypes)

train.index = train["Date"]

# Data Visualization
plot.plot(train["Close"], label= 'Close Price History')
