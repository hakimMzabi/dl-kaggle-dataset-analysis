import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.activations import *
from tensorflow.keras.datasets import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow_core.python.keras.utils import plot_model

TEST_NAME = "tf2_RNN_epochs=5"

# training parameters

BATCH_SIZE = 64
EPOCHS = 5
DROPOUT_RATE = 0.2
FILTERS = 64
NB_CLASS = 10


class PrintTrueTrainMetricsAtEpochEnd(Callback):

    def __init__(self, x_train, y_train):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.x_train, self.y_train, batch_size=1024)
        print(f"Le Vrai loss du train : {loss}")
        print(f"La Vrai acc du train : {acc}")



def create_model(k, l):
    m = Sequential()
    m.add(LSTM(100, return_sequences=True, input_shape=(k, l)))
    m.add(Dropout(0.5))
    m.add(LSTM(100, return_sequences=True, activation=relu))
    m.add(Dropout(0.5))
    m.add(LSTM(50, return_sequences=True, activation=relu))
    m.add(Dropout(0.5))
    m.add(Dense(50, activation=relu))
    # couche de sortie
    m.add(Dense(10, activation=softmax))
    m.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])

    return m


if __name__ == "__main__":
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    (x_train, y_train), (x_val, y_val) = cifar10.load_data()

    print("Before reshaping ...")

    print(x_train.shape)
    print(y_val.shape)

    y_train = tf.compat.v1.keras.utils.to_categorical(y_train)
    y_val = tf.compat.v1.keras.utils.to_categorical(y_val)

    x_train = x_train / 255.0
    x_val = x_val / 255.0

    # x_train = np.reshape(x_train, (50000, 3072, 1))
    # x_val = np.reshape(x_val, (10000, 3072, 1))

    print(x_train.shape)
    print("After reshaping ...")

    print(x_train.shape)
    print(x_val.shape)

    tensor_board_callback = TensorBoard("./logs/" + TEST_NAME)
    k = x_train.shape[1]
    l = x_train.shape[2]
    m = create_model(k, l)
    print(m.summary())
    # plot_model(m, "test_lstm.png")


    history = m.fit(x_train, y_train
                    ,epochs=EPOCHS
                    ,batch_size=2048
                    ,validation_data=(x_val, y_val))

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