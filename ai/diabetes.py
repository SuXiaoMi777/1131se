import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Input, Dense
from keras.utils import to_categorical

#顯示訓練和驗證損失的圖表
import matplotlib.pyplot as plt


np.random.seed(10) #指定亂數種子

#載入資料集
df = pd.read_csv("diabetes.csv")
dataset = df.values
np.random.shuffle(dataset) #使用亂數打亂資料

#分割為特定標籤資料
X = dataset[:, 0:8]
y = dataset[:, 8]

#特徵標準化(執行過後的模型調整)
X -= X.mean(axis=0)
X /= X.std(axis=0)

#One-hot 編碼
y = to_categorical(y)

#分割訓練和測試資料集
X_train, y_train = X[:690], y[:690]  #訓練資料前690筆
X_test, y_test = X[690:], y[690:]    #測試資料後78筆

#定義模型
model = Sequential()
model.add(Input(shape=(8,)))
model.add(Dense(10, kernel_initializer="random_uniform",
                bias_initializer="ones",
                activation="relu"))
model.add(Dense(8, kernel_initializer="random_uniform",
                bias_initializer="ones",
                activation="relu"))
model.add(Dense(2, kernel_initializer="random_uniform",
                bias_initializer="ones",
                activation="softmax"))#改啟動函數
model.summary()  #顯示模型摘要資訊

#模型編譯
model.compile(loss="binary_crossentropy", optimizer="sgd",
              metrics=["accuracy"])
#優化器=>反向傳播:sgd梯度下降

#訓練模型
history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=13, batch_size=10)         

"""
acc = history.history['loss']
epochs = range(1, len(loss)+1)
val_loss = history.history['val_loss']
plt.plot(epochs, loss, "b-" ,label= "Training Loss")
plt.plot(epochs, val_loss, "r--" ,label= "Validation Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
"""

acc = history.history['accuracy']
epochs = range(1, len(acc)+1)
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, "b-" ,label= "Training Acc")
plt.plot(epochs, val_acc, "r--" ,label= "Validation Acc")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
                            
#評估模型
loss, accuracy = model.evaluate(X_train, y_train)
print("訓練資料準確度 = {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test)
print("測試資料準確度 = {:.2f}".format(accuracy))

print(df.head())
df.head().to_html("./ch5-2-1.html")
print(df.shape)