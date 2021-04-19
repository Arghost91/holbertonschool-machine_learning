#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

bars = np.add(bars1, bars2, bars3).tolist()

names = ['Farrah', 'Fred', 'Felicia']
fruits_1 = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

#for i in range(len(fruit)):
#    plt.bar(names, fruit[i], color=colors[i], width=0.5)
    
plt.bar(names, fruit[0], bars1, color='red', label='apples', width=0.5)
plt.bar(names, fruit[0], bars2, bottom=bars1, color='yellow', label='bananas', width=0.5)
plt.bar(names, fruit[0], bars3, bottom=bars2, color='#ff8000', label='oranges', width=0.5)
plt.bar(names, fruit[0], bars4, bottom=bars2, color='#ffe5b4', label='peaches', width=0.5)


plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 90, 10))

plt.title('Number of Fruit per Person')
plt.show()
