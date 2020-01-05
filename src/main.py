from __future__ import print_function
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

ROOT = os.path.dirname(os.path.abspath(__file__))
CIFAR10_BATCHES_PATH = ROOT + (
    "\\dataset\\cifar-10-batches-py" if os.name == 'nt' else "/dataset/cifar-10-batches-py")


def is_cifar10(dataset_name):
    return dataset_name == "cifar10"


def is_fashion_mnist(dataset_name):
    return dataset_name == "fashion_mnist"


def header():
    print(f"{'=' * 60} CONFIGURATION {'=' * 60}")
    print("Project Path:", ROOT)
    print("CIFAR-10 Path:", CIFAR10_BATCHES_PATH)
    print("GPU Available:", tf.test.is_gpu_available())
    print("Building with CUDA:", tf.test.is_built_with_cuda())
    print(f"{'=' * 135}")


def show_first_samples(x_train, y_train):
    plt.imshow(x_train[0])
    print(y_train[0])
    plt.show()
    plt.imshow(x_train[1])
    print(y_train[1])
    plt.show()
    plt.imshow(x_train[2])
    print(y_train[2])
    plt.show()
    plt.imshow(x_train[3])
    print(y_train[3])
    plt.show()


def create_model(dataset_name):
    input_layer = tf.keras.layers.Input((28, 28))
    flatten_layer_output = tf.keras.layers.Flatten(name="flatten")(input_layer)

    d1 = tf.keras.layers.Dense(784, activation=tf.keras.activations.relu)(flatten_layer_output)
    d2 = tf.keras.layers.Dense(784, activation=tf.keras.activations.relu)(d1)
    d3 = tf.keras.layers.Dense(784, activation=tf.keras.activations.relu)(d2)
    d4 = tf.keras.layers.Dense(784, activation=tf.keras.activations.relu)(d3)

    d5 = tf.keras.layers.Dense(784, activation=tf.keras.activations.relu)(tf.keras.layers.Add()([d2, d4]))
    d6 = tf.keras.layers.Dense(784, activation=tf.keras.activations.relu)(tf.keras.layers.Add()([d1, d5]))

    output_tensor = tf.keras.layers.Dense(784, activation=tf.keras.activations.softmax)(
        tf.keras.layers.Add()([flatten_layer_output, d6]))
    model = tf.keras.Model(input_layer, output_tensor)

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.losses.sparse_categorical_crossentropy])
    return model

def model_fit_2():
    batch_size = 32
    num_classes = 10
    epochs = 100
    data_augmentation = True
    num_predictions = 20
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_model.h5'

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            workers=4,
                            use_multiprocessing=True
                            )

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def model_fit(dataset_name):
    if is_cifar10(dataset_name):
        dataset = cifar10
    elif is_fashion_mnist(dataset_name):
        dataset = fashion_mnist
    else:
        print("Error: No dataset name found for fitting.")
        exit()

    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train, x_test = x_train / 255, x_test / 255

    show_first_samples(x_train, y_train)
    model = create_model(dataset_name)
    # print(model.summary())
    # tf.keras.utils.plot_model(model, "unet_dense.png")
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=100,
              batch_size=8192)


header()
# model_fit("fashion_mnist")
model_fit_2()
