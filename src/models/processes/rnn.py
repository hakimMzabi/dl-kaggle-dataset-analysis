'''
RNN process to generate models for the CIFAR-10 dataset.
'''

from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.models import *
from tensorflow.keras import datasets


def create_model():
    regressor = Sequential()
    regressor.add(LSTM(50, return_sequences=True, input_dim=(32 * 32 * 3)))
    # regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(LSTM(50, return_sequences=True))
    regressor.add(LSTM(50, return_sequences=True))
    # couche de sortie
    regressor.add(Dense(units=10))
    regressor.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=[sparse_categorical_accuracy])

    return regressor


if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val) = datasets.cifar10.load_data()
    regressor = create_model()
    regressor.fit(x_train, y_train, epochs=10, batch_size=2048)
