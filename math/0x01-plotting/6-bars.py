#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

names = ['Farrah', 'Fred', 'Felicia']
fruits_1 = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

plt.bar(fruit[0], 'apples', color=colors[0], width=0.5)
plt.bar(fruit[1], 'bananas', color=colors[1], width=0.5)
plt.bar(fruit[2], 'oranges', color=colors[2], width=0.5)
plt.bar(fruit[3], 'peaches', color=colors[3], width=0.5)
plt.legend(["apples", "bananas", "oranges", "peaches"], loc="upper right")

plt.xticks(fruit, 'Farrah', 'Fred', 'Felicia')
plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 90, 10))

plt.title('Number of Fruit per Person')
