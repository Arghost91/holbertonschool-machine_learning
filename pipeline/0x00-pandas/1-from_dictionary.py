#!/usr/bin/env python3
"""
Python script that created a pd.DataFrame from a dictionary
"""
import pandas as pd


index = ['A', 'B', 'C', 'D']
diction = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ['one', 'two', 'three', 'four']
}
df = pd.DataFrame(index, data=diction)
