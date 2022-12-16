# CS5100_Project

Project for CS5100 Foundations of Artificial Intelligence at Northeastern University.	
The final project should be an application of some are of AI to a problem of interest
to you.

Group Problem Statement:

Through the use of historical data, mathematical models, and machine learning, our group 
aims to predict short term fluctuations within the New York Stock Exchange (NYSE) on a 
given day, assuming no major anomalies which impact the performance of a stock. More 
specifically, we wish to analyze stock information provided through the use of yfinance 0.1.77, 
which offers a threaded and pythonic way of downloaded market data from Yahoo Finance. The goal, 
as mentioned, is to predict the short-term future value of stocks in order to determine investment 
decisions.

Instructions for running ARIMA Code:
The best way to run the ARIMA code is to open the Jupyter notebook file (.ipynb) and run each section
seperately. However for sake of convience, a standard Python file (.py) is provided as well. Please 
note it is necessary to have all packaged listed at the top of the file installed in order to run the
model.


Instructions for running LSTM code: 
The LSTM code is located under the lstm\model directory. 
'CS5100_Project\lstm\model\lstmdatatraining.py'

Lines 10-14 are examples of the current data .csv files located under 'CS5100_Project\data' directory. 
Given the provided csv files, following the format of the "file" variable and linking it with the desired 
.csv file will automatically pull the desired data file. 

At this point nothing else needs to be done, simply run the file and watch it work. Currently the ouptut 
is set to provide a projected graph indicating the entire data set, projection data, test data and trained data.

Lines 77-80 produce the output for Y-Test matrix and Prediction Matrix and are by default commented out. 

Line 114 produces the data type output showing the value of each obj being passed through to the LSTM. 
