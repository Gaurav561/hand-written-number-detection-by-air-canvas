import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as k

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0


y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

from tensorflow.keras import layers


model = tf.keras.models.Sequential()
model.add(layers.Flatten(input_shape=(28,28)))
#model.add(layers.Dropout(0.5))
#model.add(layers.BatchNormalization())
model.add(layers.Dense(512,activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.BatchNormalization())
model.add(layers.Dense(128,activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.BatchNormalization())
#model.add(layers.Dense(64,activation='relu'))
#model.add(layers.BatchNormalization())
model.add(layers.Dense(10,activation='softmax'))


model.summary()

model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

model.fit(x_train,y_train,epochs=30,batch_size=100,validation_data=(x_test, y_test))

model.save('dense')
