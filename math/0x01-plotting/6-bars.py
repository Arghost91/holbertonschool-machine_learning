#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

names = ['Farrah', 'Fred', 'Felicia']
fruits_1 = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

for i in range(len(fruit)):
    plt.bar(names, fruit[i], color=colors[i], width=0.5)

plt.legend(["apples", "bananas", "oranges", "peaches"], loc="upper right")

plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 90, 10))

plt.title('Number of Fruit per Person')
plt.show()
