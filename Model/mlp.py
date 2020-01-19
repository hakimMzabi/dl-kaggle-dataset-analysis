import pickle as pkl
import pprint as pp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.activations import *
from tensorflow.keras.datasets import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt

CIFAR_10_BATCH_PATH = "../dataset/cifar-10-batches-py"

'''def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pkl.load(fo, encoding='bytes')
    return dict


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pkl.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels


def keys_of_cfar10(cifar10_path_folder_path, batch_id):
    with open(cifar10_path_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pkl.load(file, encoding='latin1')
    return batch['data']


def keys_of_cfar10_test(cifar10_path_folder_path):
    with open(cifar10_path_folder_path + '/test_batch', mode='rb') as file:
        batch = pkl.load(file, encoding='latin1')
    return batch['data']

# pp.pprint(unpickle("./dataset/cifar-10-python/cifar-10-batches-py/batches.meta"))
# print(unpickle("./dataset/cifar-10-python/cifar-10-batches-py/data_batch_1"))
# print(load_cfar10_batch("./dataset/cifar-10-batches-py", 1)[0])

print(keys_of_cfar10_test(CIFAR_10_BATCH_PATH).shape)
print(keys_of_cfar10_test(CIFAR_10_BATCH_PATH)[0])
'''


# ------loss: 0.0186 - categorical_accuracy: 0.8758 - val_loss: 0.0804 - val_categorical_accuracy: 0.4906---------- :
# activation 4Tanh / sigmoid; opt:sgd(lr=0.5)/ loss: mse epochs 200 #
# ----loss: 0.8592 - categorical_accuracy: 0.6889 - val_loss: 2.2219 - val_categorical_accuracy: 0.4577---: epochs:100

def create_model():
    model = Sequential()
    model.add(Dense(512, activation=relu, input_dim=32 * 32 * 3))
    model.add(Dense(256, activation=relu))
    model.add(Dense(128, activation=relu))
    model.add(Dense(64, activation=relu))
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

    x_train = np.reshape(x_train, (-1, 32 * 32 * 3)) / 255.0
    x_val = np.reshape(x_val, (-1, 32 * 32 * 3)) / 255.0
    print(x_train)

    print("After reshaping ...")

    print(x_train.shape)
    print(x_val.shape)

    print("ici", np.asarray(y_train))

    m = create_model()
    print(m.summary())

    m.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, batch_size=1024)
