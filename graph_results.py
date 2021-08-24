import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

with open('3cars3routes_SLSQP.csv', 'rb') as f:
    results = np.loadtxt(f, delimiter = ',')

x = np.linspace(1,10,10)
y = results.mean(axis=0)
plt.plot(x,y)
plt.scatter(x,y)
plt.show()