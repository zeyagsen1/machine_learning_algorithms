import numpy as np
from matplotlib import pyplot as plt


standart_deviation=1
mean=2
data1 = np.random.uniform(0, 10, size=1000)
data2 = np.random.normal(mean, standart_deviation, size=1000)

np.random.seed(42)
plt.hist(data1, bins=30, density=True, color='skyblue', edgecolor='black')
plt.title('Uniform Distribution [0, 10)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()  

plt.hist(data2, bins=30, density=True, color='lightgreen', edgecolor='black')
plt.title('Normal Distribution (Mean = 2, Std = 1)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show() 

