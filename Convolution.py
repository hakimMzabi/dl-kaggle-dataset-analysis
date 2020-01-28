from tensorflow.keras.activations import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import datasets, layers, models

(x_train, y_train), (x_val, y_val) = datasets.cifar10.load_data()
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation=relu, input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
#couche complétement connectée
model.add(layers.Dense(128, activation='relu'))
#couche de sortie
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)