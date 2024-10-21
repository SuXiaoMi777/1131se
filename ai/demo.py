import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+(np.e**(-x)))

def derivartive(x):
    return sigmoid(x) *(1 - sigmoid(x)) 

x = np.arange(-6,6,0.1)

plt.plot(x,derivartive(x))
plt.title("derivartive of sigmoid function")
plt.show()