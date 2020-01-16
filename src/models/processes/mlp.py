'''
Multi-Layer Perceptron process to generate models for the CIFAR-10 dataset.
'''

from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import *
from src.helper import Helper


def create_model_1():
    model_1 = Sequential()

    model_1.add(Dense(96, activation=relu, input_dim=3072))  # 32 * 32 * 3
    model_1.add(Dropout(0.1))
    model_1.add(Dense(48, activation=relu))
    model_1.add(Dropout(0.1))
    model_1.add(Dense(10, activation=softmax))

    model_1.compile(
        optimizer=Adam(),
        loss=sparse_categorical_crossentropy,
        metrics=[sparse_categorical_accuracy]
    )

    return model_1


if __name__ == '__main__':
    # Initialize the helper
    helper = Helper()

    # Load dataset
    (x_train, y_train), (x_test, y_test) = helper.get_cifar10_prepared()

    model = create_model_1()
    helper.save_model(model, "mlp")
    model_loaded = helper.load_model(helper.get_models_last_filename("mlp"))
    model_loaded.summary()
    helper.fit(
        model,
        x_train,
        y_train,
        batch_size=1024,
        epochs=10,
        validation_data=(x_test, y_test),
        process_name="mlp"
    )
