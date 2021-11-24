#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.drop(columns='Weighted_Price')

df = df.fillna({'High': df['Close'].shift(periods=1, fill_value=0),
                'Low': df['Close'].shift(periods=1, fill_value=0),
                'Open': df['Close'].shift(periods=1, fill_value=0),
                'Close': df['Close'].shift(periods=1, fill_value=0)})

df["Volume_(BTC)"].fillna(value=0, inplace=True)
df["Volume_(Currency)"].fillna(value=0, inplace=True)

print(df.head())
print(df.tail())
