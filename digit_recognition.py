import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train_flattened = x_train.reshape(len(x_train),28*28)
x_test_flattened = x_test.reshape(len(x_test),28*28)
# print(x_train_flattened.shape)

x_train_flattened = x_train_flattened/255
x_test_flattened = x_test_flattened/255

# random_image = x_test_flattened[0].reshape(28,28)
# plt.imshow(random_image, cmap = matplotlib.cm.binary)
# print(y_train[0])


model = keras.Sequential([keras.layers.Dense(10, input_shape=(784,),activation = 'sigmoid')])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'] )

model.fit(x_train_flattened, y_train, epochs = 5)
model.evaluate(x_test_flattened,y_test)

# y_pred = model.predict(x_test_flattened)
# y_pred_labels = [np.argmax(i) for i in y_pred]
# print(y_pred_labels[0:50])
# print(y_test[0:50])
