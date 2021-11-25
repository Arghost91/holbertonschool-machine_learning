#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df = df.set_index('Date')
df['Close'].fillna(method='ffill', inplace=True)
df = df.fillna({'High': df['Close'].shift(periods=1, fill_value=0),
                'Low': df['Close'].shift(periods=1, fill_value=0),
                'Open': df['Close'].shift(periods=1, fill_value=0)})

df['Volume_(BTC)'].fillna(value=0, inplace=True)
df['Volume_(Currency)'].fillna(value=0, inplace=True)
df = df[(df.index >= '2017-01-01')]
df = df.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min',
                          'Close': 'last', 'Volume_(BTC)': 'sum',
                          'Volume_(Currency)': 'sum'})
df = df.drop(['Weighted_Price'], axis=1)

df.plot()
plt.show()
