'''
Multi-Layer Perceptron process to generate models for the CIFAR-10 dataset.
'''

from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import *
from tensorflow.keras.datasets import cifar10
from src.helper import Helper


def create_model_1():
    model = Sequential()

    model.add(Dense(96, activation=relu, input_dim=3072))  # 32 * 32 * 3
    model.add(Dropout(0.1))
    model.add(Dense(48, activation=relu))
    model.add(Dropout(0.1))
    model.add(Dense(24, activation=relu))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation=softmax))

    model.compile(
        optimizer=Adam(),
        loss=sparse_categorical_crossentropy,
        metrics=[sparse_categorical_accuracy]
    )

    return model


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #
    # # Normalize the data
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0
    #
    # x_train = x_train.reshape((50000, 32 * 32 * 3))
    # x_test = x_test.reshape((10000, 32 * 32 * 3))
    #
    # model = create_model_1()
    # model.summary()
    # model.fit(
    #     x_train,
    #     y_train,
    #     batch_size=1024,
    #     epochs=100,
    #     validation_data=(x_test, y_test)
    # )

    # Initialize the helper
    helper = Helper()

    model = create_model_1()
    helper.save_model(model, "MLP")
