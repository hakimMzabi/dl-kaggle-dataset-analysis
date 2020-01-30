'''
ConvNet process to generate models for the CIFAR-10 dataset.
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.activations import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

TEST_NAME = "tf2_convnet_mlp_epochs=100"

# ---3s 66us/sample - loss: 0.3325 - categorical_accuracy: 0.8829 - val_loss: 0.5924 - val_categorical_accuracy: 0.8178 BEST
# -----loss: 0.3161 - categorical_accuracy: 0.8874 - val_loss: 0.7152 - val_categorical_accuracy: 0.7836
# 10000/1 - 1s - loss: 0.4760 - categorical_accuracy: 0.7836 : epochs 100; batch size : 64
# ------loss: 0.0186 - categorical_accuracy: 0.8458 - val_loss: 0.0804 - val_categorical_accuracy: 0.7406---------- :
# activation 4 Relu / softmax; opt:Adam/ loss: categorical_accuracy epochs 100, batch size: 2048#
# ----loss: 0.8592 - categorical_accuracy: 0.6889 - val_loss: 2.2219 - val_categorical_accuracy: 0.4577---: epochs:100


# training parameters
BATCH_SIZE = 1024
EPOCHS = 250
DROPOUT_RATE = 0.2
FILTERS = 64
NB_CLASS = 10


def create_model():
    model = Sequential()
    model.add(Conv2D(FILTERS, (3, 3), padding='same', activation=relu, input_shape=(32, 32, 3)))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Conv2D(FILTERS, (3, 3), padding='same', activation=relu))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same', activation=relu))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Conv2D(FILTERS, (3, 3), padding='same', activation=relu))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Flatten())
    model.add(Dense(512, activation=relu))
    model.add(Dropout(0.5))
    model.add(Dense(NB_CLASS, activation=softmax))

    model.compile(optimizer=Adam(),
                  loss=categorical_crossentropy,
                  metrics=[categorical_accuracy])

    return model


def create_model_2():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10,
                        validation_data=(x_test, y_test))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print(test_acc)


def create_model_3():
    (x_train, y_train), (x_val, y_val) = datasets.cifar10.load_data()
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation=relu, input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    # couche complétement connectée
    model.add(layers.Dense(128, activation='relu'))
    # couche de sortie
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)


if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val) = cifar10.load_data()

    print("Before reshaping ...")

    print(x_train.shape)
    print(y_val.shape)

    y_train = tf.compat.v1.keras.utils.to_categorical(y_train)
    y_val = tf.compat.v1.keras.utils.to_categorical(y_val)

    x_train = x_train / 255.0
    x_val = x_val / 255.0

    print("After reshaping ...")

    print(x_train.shape)
    print(x_val.shape)

    # tensor_board_callback = TensorBoard("./logs/" + TEST_NAME)

    m = create_model()
    print(m.summary())
    print(y_val[4])

    history = m.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))
    # callbacks=[tensor_board_callback])

    plt.plot(history.history['categorical_accuracy'], label='categorical_accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='val_categorical_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = m.evaluate(x_val, y_val, verbose=2)
    print(test_acc)
    # Testing predictions
    predict = m.predict(x_val)
    print(predict[0])
    print(np.argmax(predict[0]))

    '''test = np.array([1.0844562e-05, 2.4671457e-05, 6.7017581e-06, 9.9533951e-01, 3.6256204e-06,
                     4.5674662e-03, 1.9694484e-05, 8.8469806e-06, 1.2012905e-05, 6.6825801e-06]) * 255.0'''
