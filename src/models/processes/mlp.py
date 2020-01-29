'''
Multi-Layer Perceptron process to generate models for the CIFAR-10 dataset.
'''

from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import *
from src.helper import Helper


def create_model(optimizer=None, dropout_values=None, activation=relu):
    model = Sequential()

    model.add(Dense(128, activation=activation, input_dim=3072))  # 32 * 32 * 3
    if dropout_values is not None:
        model.add(Dropout(dropout_values[0]))
    model.add(Dense(64, activation=activation))
    if dropout_values is not None:
        model.add(Dropout(dropout_values[1]))
    model.add(Dense(10, activation=softmax))

    model.compile(
        optimizer=optimizer.lower(),
        loss=sparse_categorical_crossentropy,
        metrics=[sparse_categorical_accuracy]
    )

    return model


if __name__ == '__main__':
    # Initialize the helper
    helper = Helper()

    # Load dataset
    (x_train, y_train), (x_test, y_test) = helper.get_cifar10_prepared()

    model = create_model(optimizer="Adam", dropout_values=[0.1, 0.2], activation=relu)
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
