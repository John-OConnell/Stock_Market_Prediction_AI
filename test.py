import pandas as pd

aapl_data = pd.read_csv('aapl_data.csv')
aapl_data = aapl_data[['Close']]


xom_data = pd.read_csv('xom_data.csv')
xom_data = xom_data[['Close']]

jnj_data = pd.read_csv('jnj_data.csv')
jnj_data = jnj_data[['Close']]

sp500_data = pd.read_csv('sp500_data.csv')
sp500_data = sp500_data[['Close']]
