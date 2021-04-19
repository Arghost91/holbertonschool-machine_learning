#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

plt.bar(fruit, 'apples', color='red', width=0.5)
plt.bar(fruit, 'bananas', color='yellow', width=0.5)
plt.bar(fruit, 'oranges', color='#ff8000', width=0.5)
plt.bar(fruit, 'peaches', color='#ffe5b4', width=0.5)
plt.legend(["apples", "bananas", "oranges", "peaches"], loc="upper right")

plt.xticks(fruit, 'Farrah', 'Fred', 'Felicia')
plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 90, 10))

plt.title('Number of Fruit per Person')
