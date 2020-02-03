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
import matplotlib.pyplot as plt

# ------loss: 0.0186 - categorical_accuracy: 0.8758 - val_loss: 0.0804 - val_categorical_accuracy: 0.4906---------- :
# batch size 1024 epochs 200 #

# ----loss: 0.8592 - categorical_accuracy: 0.6889 - val_loss: 2.2219 - val_categorical_accuracy: 0.4577---: epochs:100

# training parameters
BATCH_SIZE = 1024
EPOCHS = 200
NB_CLASS = 10
TEST_NAME = "tf2_mlp_epochs=100_batch_size=1024"

def create_model():

    model = Sequential()
    model.add(Dense(512, activation=relu, input_dim=32* 32* 3))
    model.add(Dense(256, activation=relu))
    model.add(Dense(128, activation=relu))
    model.add(Dense(64, activation=relu))

    model.add(Dense(NB_CLASS, activation=softmax))

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

    x_train = np.reshape(x_train, (-1, 32 * 32 * 3)) / 255.0
    x_val = np.reshape(x_val, (-1, 32 * 32 * 3)) / 255.0
    print(x_train)

    print("After reshaping ...")

    print(x_train.shape)
    print(x_val.shape)

    print("ici", np.asarray(y_train))

    # tensor_board_callback = TensorBoard("./logs/" + TEST_NAME)

    m = create_model()
    print(m.summary())

    history = m.fit(x_train, y_train
                    ,validation_data=(x_val, y_val)
                    ,epochs=EPOCHS
                    ,batch_size=BATCH_SIZE)
                    #,callbacks=[tensor_board_callback])

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

