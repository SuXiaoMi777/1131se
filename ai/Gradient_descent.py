import numpy as np
import matplotlib.pyplot as plt 

def L(w):
    return w * w
def dL(w):
    return 2 * w
def gradient_descent(w_start, dL, lr, epochs):
            #初始w值:從何處開始梯度下降
            #lr Learning Rate:步距
            #epochs:迭代次數:梯度下降將進行多少次
    w_gd = []
    w_gd.append(w_start)
    pre_w = w_start
    
    for i in range(epochs):
        w = pre_w - lr * dL(pre_w)
        w_gd.append(w)
        pre_w = w
    return np.array(w_gd)
    
w0 = 5
epochs = 5
lr = 0.4
w_gd = gradient_descent(w0, dL, lr, epochs)

print(w_gd)

t = np.arange(-5.5, 5.5, 0.01)
plt.plot(t, L(t), c='b') #color = blue
plt.plot(w_gd, L(w_gd), c='r', label='lr={}'.format(lr))
plt.scatter(w_gd, L(w_gd), c='r')
plt.legend()
plt.show()