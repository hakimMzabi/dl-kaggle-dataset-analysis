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

from src.cifar10 import Cifar10


def create_model(
        optimizer="Adam",
        dropout_values=None,
        activation=relu,
        filter_size=64,
        padding_values="same",
        kernel_size=(3, 3),
        max_pool_values=None
):
    model = Sequential()

    if len(dropout_values) != len(padding_values):
        print("Number of dropout values must be equal to number of padding values")
        return

    for i in range(len(dropout_values)):
        if i == 0:
            model.add(Conv2D(filter_size, kernel_size, padding='same', activation=activation, input_shape=(32, 32, 3)))
        else:
            model.add(Conv2D(filter_size, kernel_size, padding='same', activation=activation))
            model.add(MaxPool2D(max_pool_values[i][0], max_pool_values[i][1]))
            model.add(Dropout(dropout_values[i]))

    model.add(Flatten())
    model.add(Dense(256, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation=softmax))

    model.compile(optimizer=optimizer,
                  loss=sparse_categorical_crossentropy,
                  metrics=[sparse_categorical_accuracy])

    return model


if __name__ == "__main__":
    cifar10 = Cifar10(dim=3)

    model = create_model(
        "Adam",
        dropout_values=[0.5, 0.5, 0.5, 0.5],
        activation=relu,
        filter_size=64,
        padding_values="same",
        kernel_size=(3, 3),
        max_pool_values=[
            [2, 2],
            [2, 2],
            [2, 2],
            [2, 2]
        ]
    )
    model.build()
    model.summary()
    cifar10.helper.fit(
        model, cifar10.x_train, cifar10.y_train, 1024, 100, validation_data=(cifar10.x_test, cifar10.y_test),
        process_name="convnet"
    )
