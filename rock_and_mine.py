import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split

sonar = pd.read_csv('sonar.csv')
print(sonar.head())
print(sonar.shape)
y = sonar['Class']

y = y.map({
    'Rock':0,
    'Mine':1
})

X = sonar.drop(['Class'], axis=1)

y_dummies = y
y = pd.get_dummies(y_dummies)
print(X)
print(y)
print(X.shape)
print(y.shape)
x_train,  x_test,y_train, y_test = train_test_split(X,y,test_size=0.3)

# # print(x_train.head())
# model = keras.Sequential([keras.layers.Dense(2, input_dim = 60,activation = 'sigmoid')])

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
MinMaxScaler(copy=True,feature_range=(0,1))
scalar.fit(x_train)

scaled_X_train = scalar.transform(x_train)
scaled_X_test = scalar.transform(x_test)
# print(scaled_X_train.shape)
# print(scaled_X_test.shape)



from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(8,input_dim = 60,activation='relu'))
model.add(Dense(8,input_dim = 60,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )
model.fit(scaled_X_train, y_train, epochs = 300)
model.evaluate(scaled_X_test,y_test)
y_pred = (model.predict(x_test) > 0.5).astype("int32")
# print(y_pred)
