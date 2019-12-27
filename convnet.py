import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.activations import *
from tensorflow.keras.datasets import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt

TEST_NAME = "tf2_convnet_mlp_epochs=100"


# -----loss: 0.3161 - categorical_accuracy: 0.8874 - val_loss: 0.7152 - val_categorical_accuracy: 0.7836
# 10000/1 - 1s - loss: 0.4760 - categorical_accuracy: 0.7836 : epochs 100; batch size : 64
# ------loss: 0.0186 - categorical_accuracy: 0.8458 - val_loss: 0.0804 - val_categorical_accuracy: 0.7406---------- :
# activation 4 Relu / softmax; opt:Adam/ loss: categorical_accuracy epochs 100, batch size: 2048#
# ----loss: 0.8592 - categorical_accuracy: 0.6889 - val_loss: 2.2219 - val_categorical_accuracy: 0.4577---: epochs:100


# loss: 0.1512 - categorical_accuracy: 0.9449 - val_loss: 1.5592 - val_categorical_accuracy: 0.7188 :
# batch_size:64/epochs :30
# After drop out loss: 0.4732 - categorical_accuracy: 0.8324 - val_loss: 0.6776 -
# val_categorical_accuracy: 0.7709

def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation=relu, input_shape=(32, 32, 3)))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same', activation=relu))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same', activation=relu))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same', activation=relu))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(10, activation=softmax))

    model.compile(optimizer=Adam(),
                  loss=categorical_crossentropy,
                  metrics=[categorical_accuracy])

    return model


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

    history = m.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val))
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