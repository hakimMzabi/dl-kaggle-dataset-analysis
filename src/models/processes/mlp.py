'''
Multi-Layer Perceptron process to generate models for the CIFAR-10 dataset.
'''

from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import *
from src.helper import Helper
from src.reporter import Reporter


def create_model_1():
    model = Sequential()

    model.add(Dense(96, activation=relu, input_dim=3072))  # 32 * 32 * 3
    model.add(Dropout(0.1))
    model.add(Dense(48, activation=relu))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation=softmax))

    model.compile(
        optimizer=Adam(),
        loss=sparse_categorical_crossentropy,
        metrics=[sparse_categorical_accuracy]
    )

    return model


if __name__ == '__main__':
    # Initialize the helper
    helper = Helper()

    # Load dataset
    (x_train, y_train), (x_test, y_test) = helper.get_cifar10_prepared()

    model = create_model_1()
    helper.save_model(model, "MLP")
    model_loaded = helper.load_model(helper.get_models_last_filename("MLP"))
    model_loaded.summary()
    model.fit(
        x_train,
        y_train,
        batch_size=1024,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=[Reporter(x_train, y_train, 1024, helper.get_models_last_filename_to_generate())]
    )


