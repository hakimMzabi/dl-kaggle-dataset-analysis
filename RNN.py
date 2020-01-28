import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.activations import *
from tensorflow.keras.datasets import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from tensorflow.keras import datasets
(x_train, y_train), (x_val, y_val) = datasets.cifar10.load_data()
regressor = Sequential()
regressor.add(LSTM(50, return_sequences=True,  input_dim=(32*32*3)))
#regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(LSTM(50, return_sequences=True))
regressor.add(LSTM(50, return_sequences=True))
#couche de sortie
regressor.add(Dense(units=10))
regressor.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=[sparse_categorical_accuracy])
regressor.fit(x_train, y_train, epochs=10, batch_size=2048)