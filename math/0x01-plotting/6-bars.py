#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

names = ['Farrah', 'Fred', 'Felicia']
fruits_1 = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

plt.bar(names, fruit[0], color='red', label='apples', width=0.5)
plt.bar(names, fruit[1], bottom=(fruit[0]), color='yellow', label='bananas', width=0.5)
plt.bar(names, fruit[2], bottom=(fruit[0]+fruit[1]), color='#ff8000', label='oranges', width=0.5)
plt.bar(names, fruit[3], bottom=(fruit[0]+fruit[1]+fruit[2]), color='#ffe5b4', label='peaches', width=0.5)

plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 90, 10))

plt.title('Number of Fruit per Person')
plt.legend()
plt.show()
