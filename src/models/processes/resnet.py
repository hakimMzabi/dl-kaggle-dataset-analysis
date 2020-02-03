'''
ResNet process to generate models for the CIFAR-10 dataset.
'''
import tensorflow as tf
from tensorflow.keras.activations import *
from tensorflow.keras.datasets import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

# training parameters
BATCH_SIZE = 1024
EPOCHS = 50
DROPOUT_RATE = 0.2
FILTERS = 64
NB_CLASS = 10


def residual_layer(inputs,
                   filters=FILTERS,
                   kernel_size=3,
                   activation=relu,
                   strides=1,
                   batch_norm=True,
                   conf_first=True):
    conv = Conv2D(filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  kernel_initializer='he_normal',
                  padding='same')
    x = inputs
    if conf_first:
        x = conv(x)
        x = Dropout(DROPOUT_RATE)(x)
    else:
        if batch_norm:
            x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
    return x


def resnet_model(input_shape, depth, num_classes=NB_CLASS):
    filter = 16
    num_res_layer = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = residual_layer(inputs=inputs)
    for stack in range(3):
        for i in range(num_res_layer):
            strides = 1

            if stack > 0 and i == 0:
                strides = 2

            y = residual_layer(inputs=x,
                               filters=filter,
                               strides=strides)

            y = residual_layer(inputs=y,
                               filters=filter,
                               activation=None)

            if strides > 0 and i == 0:
                x = residual_layer(inputs=x,
                                   filters=filter,
                                   kernel_size=1,
                                   strides=strides,
                                   activation=None,
                                   batch_norm=False)
            layer_name = f"add_dense_{stack}_{i}"
            print(layer_name)
            x = add([x, y], name=layer_name)
            x = Activation(relu)(x)
        filter = filter * 2

    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)

    output = Dense(num_classes, activation=softmax, kernel_initializer='he_normal', name=f"dense_output")(y)
    model = Model(inputs=inputs, outputs=output)

    return model


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()

    (x_train, y_train), (x_val, y_val) = cifar10.load_data()

    input_shape = x_train.shape[1:]
    depth = 3 * 6 + 2

    x_train = x_train / 255.0
    x_val = x_val / 255.0

    m = resnet_model(input_shape, depth)

    # Compilation du modèle
    m.compile(Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    m.summary()
    # plot_model(m)

    m.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)
