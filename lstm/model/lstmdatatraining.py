
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plot
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
# file = "sp500_data.csv"
file = "aapl_data.csv"
# file = "jnj_data.csv"
# file = "xom_data.csv"

'''
Building LSTM model requires separation of stock price data into a trianing set and a test set
This requires normalizing the data so that the values range from 0 to 1. 
Instead of taking the entirety of that data, we will focus the extraction of 80% of the data set as test set.
'''
def trainingSet(stockData):
    closePrice = stockData['Close']
    closeValues = closePrice.values
    trainingLen = math.ceil(len(closeValues)* 0.8)

    # MinMax Scaler: renomarlized into 2d array
    minMax = MinMaxScaler(feature_range=(0,1))
    scaleData = minMax.fit_transform(closeValues.reshape(-1,1))
    trainData = scaleData[0:trainingLen, :]

    xTrain, yTrain = [], []
    for i in range(60, len(trainData)):
        xTrain.append(trainData[i-60:i, 0])
        yTrain.append(trainData[i, 0])
    
    '''
    Convertaion featureData-> Xtrain and labelDate -> ytrain into 
    Numpy array as it is the data valid through tensorflow
    Then LSTM model requires a 3d array
    '''
    xTrain = np.array(xTrain) 
    yTrain = np.array(yTrain)
    xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

    testData = scaleData[trainingLen-60: , :]
    xTest = []
    yTest = closeValues[trainingLen:]

    for i in range(60, len(testData)):
        xTest.append(testData[i-60:i,0])
    
    xTest = np.array(xTest)
    xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1],1))

    return xTrain, yTrain, xTest, yTest, minMax, trainingLen

'''
Setting up LSTM network
Model comprised of linear stack of layers, LSTM 100 network units, the return sequence will be same length

'''
def LSTMmodel(xTrain, yTrain, xTest, yTest, minMax):
    lstmModel = tf.keras.Sequential()
    lstmModel.add(layers.LSTM(100, return_sequences=True, input_shape=(xTrain.shape[1], 1)))
    lstmModel.add(layers.LSTM(100, return_sequences=False))
    lstmModel.add(layers.Dense(25))
    lstmModel.add(layers.Dense(1))
    lstmModel.summary()

    # Training
    lstmModel.compile(optimizer='adam', loss='mean_squared_error')
    lstmModel.fit(xTrain, yTrain, batch_size=1, epochs=3)

    prediction = lstmModel.predict(xTest)
    prediction = minMax.inverse_transform(prediction)
    #RMSE = np.sqrt(np.mean(prediction - yTest) ** 2)
    # mse = np.square(np.subtract(yTest, prediction)).mean()
    # rmse = math.sqrt(mse)
    # print(rmse)
    # print(RMSE)

    print('************************YTEST************************\n')
    print(yTest)
    print('************************PREDICTION************************\n')
    print(prediction)

    return prediction
    #, RMSE

def visualizeOutput(stockData, prediction, trainingLen):
    data = stockData.filter(['Close'])
    train = data[:trainingLen]
    validate = data[trainingLen:]
    validate['Predictions'] = prediction
    plot.figure(figsize=(16,8))
    plot.title('LSTM Model')
    plot.xlabel('Date')
    plot.ylabel('Closing Price USD$')
    plot.plot(train)
    plot.plot(validate[['Close', 'Predictions']])
    plot.legend(['Train', 'val', 'Predictions'], loc='lower right')
    plot.show()
    

def main():
    # reading single file as variable through the above attribute
    train = pd.read_csv(f'CS5100_Project\data\{file}')

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
    # plot.plot(train["Close"], label= 'Close Price History')

    # plot.figure(figsize=(15,8))
    # plot.title('Stock Prices History')
    # plot.plot(train['Close'])
    # plot.xlabel('Date')
    # plot.ylabel('Prices ($)')
    # plot.show()

    # plot.plot(train["Close"],label='Close Price history')

    # create trainingset
    xTrain, yTrain, xTest, yTest, minMax, trainingLen = trainingSet(train)
    prediction = LSTMmodel(xTrain, yTrain, xTest, xTest, minMax)

    visualizeOutput(train, prediction, trainingLen)

if __name__ == "__main__":
    main()
