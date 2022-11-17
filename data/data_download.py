import yfinance as yf
import pandas as pd


# download necessary stock data and save to csv files
aapl_data = yf.download('AAPL', start='2002-11-01', end='2022-11-01')
aapl_data = pd.DataFrame(aapl_data)
aapl_data.to_csv('aapl_data.csv')

xom_data = yf.download('XOM', start='2002-11-01', end='2022-11-01')
xom_data = pd.DataFrame(xom_data)
xom_data.to_csv('xom_data.csv')

jnj_data = yf.download('JNJ', start='2002-11-01', end='2022-11-01')
jnj_data = pd.DataFrame(jnj_data)
jnj_data.to_csv('jnj_data.csv')

sp500_data = yf.download('^GSPC', start='2002-11-01', end='2022-11-01')
sp500_data = pd.DataFrame(sp500_data)
sp500_data.to_csv('sp500_data.csv')
