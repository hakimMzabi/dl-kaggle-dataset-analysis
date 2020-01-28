import os
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

from src.reporter import Reporter
from tensorflow.keras.datasets import cifar10


class Helper:
    """
    Helper class is here to help the developer debugging machine learning resources and variables and manage processes.
    """

    def __init__(self):
        self.src_path = os.path.dirname(os.path.realpath(__file__))
        self.models_responses_folder = self.src_path + "\\models\\responses\\"
        self.checkpoint_folder = self.src_path + "\\checkpoints\\"


    def get_models_last_filename(self, process_name) -> str:
        """
        Returns the last filename of the models generated returns "No model generated." if there is no file.
        :param process_name: mlp, resnet, rnn...
        :return: the last model generated (e.g if mlp_42.h5 is the file with the highest id, it returns mlp_42.h5)
        """
        models_last_num = self.get_models_last_num(process_name)
        return self.get_model_filename(process_name, models_last_num) if models_last_num != -1 else \
            process_name + "_1"

    def get_models_last_num(self, process_name) -> int:
        """
        Return the last id of the models generated in src/models/response or -1 if there is no file
        :param process_name: mlp, resnet, rnn...
        :return: the max number of the model (e.g if mlp_42.h5 is the file with the highest id, it returns 42)
        """
        last_num_to_generate = int(self.get_models_last_num_to_generate(process_name))
        return last_num_to_generate - 1 if last_num_to_generate - 1 != 0 else -1

    def get_model_filename(self, process_name, num) -> str:
        """
        Get a model filename from a number specified
        :param process_name: mlp, resnet, rnn...
        :param num: id to select for the selected process_name
        :return: the full filename for the selected process_name and num
        """
        return self.models_responses_folder + process_name.lower() + "_" + str(num) + ".h5" \
            if os.path.isdir(self.models_responses_folder) \
            else "Couldn't get a model filename"

    def get_models_last_filename_to_generate(self, process_name) -> str:
        """
        Get the last model filename to generate in the src/models/responses folder
        :param process_name: mlp, resnet, rnn...
        :return: the filename with the highest id + 1 of the models (e.g if mlp_42.h5 is the file with the highest id,
        it returns mlp_43.h5)
        """
        return self.get_model_filename(process_name, self.get_models_last_num_to_generate(process_name))

    def get_models_last_num_to_generate(self, process_name) -> int:
        """
        Get the last number of the model to generates to handle filename incrementation
        :param process_name: mlp, resnet, rnn...
        :return: the max number of the model (e.g if mlp_42.h5 is the file with the highest id, it returns 43)
        """
        if os.path.isdir(self.models_responses_folder):
            max = 0
            for (dirpath, dirnames, filenames) in os.walk(self.models_responses_folder):
                for filename in filenames:
                    if process_name.lower() in filename:
                        number = int(filename.split(process_name.lower() + "_")[1].split('.')[0])
                        if number > max:
                            max = number
            return max + 1
        print(f"Error : The directory {self.models_responses_folder} does not exists")
        exit()

    def save_model(self, model, process_name):
        """
        Saves the model in the src/models/responses folder, automatically increments from the last model created
        :param model: tensorflow keras model
        :param process_name: mlp, resnet
        :return: nothing
        """
        model.save(self.get_models_last_filename_to_generate(process_name))

    @staticmethod
    def show_samples(x_train, y_train):
        """
        # Show samples of an image dataset
        :param x_train: features
        :param y_train: labels
        :return: nothing
        """
        for i in range(10):
            plt.imshow(x_train[i])
            print(y_train[i])
            plt.show()

    @staticmethod
    def create_dir(path):
        """
        Creates a directory if it doesn't exist and print an error if it is not possible
        :param path: e.g "/random_dir/the_new_dir/
        :return: nothing
        """
        try:
            if not os.path.isdir(path):
                os.mkdir(path)
        except OSError:
            print(f"Couldn't create the dir : {path}")
        else:
            print(f"Successfully created the dir : {path}")

    @staticmethod
    def list_to_str_semic(list) -> str:
        res = "["
        for el in list:
            res += str(el) + ";"
        res += "]"
        return res

    @staticmethod
    def debug_dataset_shapes(dataset_name, dataset, terminate=False):
        """
        Show dataset shapes
        Dataset must be equal to [x_train, y_train, x_test, y_test]
        :param dataset_name: e.g "cifar10"
        :param dataset: [x_train, y_train, x_test, y_test]
        :param terminate: True if you want the program to terminates, False otherwise
        :return:
        """
        print(f"[DEBUGGER] Debugging the {dataset_name} dataset :")
        print(f"[DEBUGGER]     x_train shape : {dataset[0].shape}")
        print(f"[DEBUGGER]     y_train shape : {dataset[1].shape}")
        print(f"[DEBUGGER]     x_test shape : {dataset[2].shape}")
        print(f"[DEBUGGER]     y_test shape : {dataset[3].shape}")
        if terminate:
            exit()

    @staticmethod
    def get_cifar10_prepared() -> (tuple, tuple):
        """
        Returns the cifar10 dataset normalized and well shaped for training as 2 tuples of 4 tensors
        :return: (tuple1 : 2 training tensors of features and labels, tuple2 : 2 validation tensors of
        features and labels)
        """
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # Normalize the data
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        # Reshape the data for training
        x_train = x_train.reshape((50000, 32 * 32 * 3))
        x_test = x_test.reshape((10000, 32 * 32 * 3))
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def load_model(model_filename=None):
        """
        Returns a tensorflow keras model from the filename
        :param model_filename
        :return: a tensorflow keras model
        """
        try:
            return tf.keras.models.load_model(model_filename)
        except FileNotFoundError:
            print("Error: Couldn't load the model. Check if the file exists.")

    @staticmethod
    def create_file(path):
        print(path)
        f = open(f"{path}", "w")
        f.close()

    def fit(self, model, x_train, y_train, batch_size, epochs, validation_data, process_name):
        """
        Fit a model and adds a checkpoint to avoid losing data in case of failure.
        Checkpoint is also useful in case of overfitting
        :param model: tensorflow keras model
        :param x_train: features
        :param y_train: labels
        :param batch_size:
        :param epochs:
        :param validation_data: test features and test labels
        :param process_name: mlp, convnet, resnet...
        :return: nothing
        """

        self.save_model(model, process_name)

        (x_test, y_test) = validation_data

        model_name = self.get_models_last_filename(process_name).split("\\")[-1].replace(".h5", "")
        log_file_path = self.src_path + "\\models\\logs\\" + model_name + ".log"
        checkpoint_file_path = self.src_path + "\\models\\checkpoints\\" + model_name + ".ckpt"
        tensorboard_log_dir = self.src_path + "\\models\\logs\\tensorboard_" + model_name + "\\fit\\" + datetime.datetime.now()\
            .strftime("%Y%m%d-%H%M%S")

        self.create_file(log_file_path)
        self.create_file(checkpoint_file_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_file_path,
            save_weights_only=True,
            verbose=1
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=[
                Reporter(x_train, y_train, batch_size, model_name, log_file_path),
                cp_callback,
                tensorboard_callback
            ]
        )
