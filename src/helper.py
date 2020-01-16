import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10


class Helper:
    """
    Helper class is here to help the developer debugging machine learning resources and variables and manage processes.
    """

    def __init__(self):
        self.accepted_processes = ["mlp", "convnet", "resnet", "rnn"]
        self.models_folder = "../responses/"

    def process_name_is_valid(self, process_name) -> bool:
        """
        Check if the process name is equal to "mlp","convnet","resnet" or rnn
        :param process_name: mlp, resnet, rnn...
        :return: True if the process_name is in the accepted_processes, else False
        """
        return process_name.lower() in self.accepted_processes

    def get_models_last_filename(self, process_name) -> str:
        """
        Returns the last filename of the models generated returns "No model generated." if there is no file.
        :param process_name: mlp, resnet, rnn...
        :return: the last model generated (e.g if mlp_42.h5 is the file with the highest id, it returns mlp_42.h5)
        """
        models_last_num = self.get_models_last_num(process_name)
        return self.get_model_filename(process_name, models_last_num) if models_last_num != -1 else \
            "No model found in the \"src/models/responses\" folder."

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
        return self.models_folder + process_name.lower() + "_" + str(num) + ".h5" \
            if self.process_name_is_valid(process_name) and os.path.isdir(self.models_folder) \
            else "Couldn't get a valid model filename"

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
        if os.path.isdir(self.models_folder):
            max = 0
            for (dirpath, dirnames, filenames) in os.walk(self.models_folder):
                for filename in filenames:
                    if process_name.lower() in filename:
                        number = int(filename.split(process_name.lower() + "_")[1].split('.')[0])
                        if number > max:
                            max = number
            return max + 1
        print(f"Error : The directory {self.models_folder} does not exists")
        exit()

    def save_model(self, model, process_name):
        """
        Saves the model in the src/models/responses folder, automatically increments from the last model created
        :param model: tensorflow keras model
        :param process_name: mlp, resnet
        :return: nothing
        """
        model_filename = ""
        if self.process_name_is_valid(process_name):
            print(f"Generating save : {model_filename}...", end="")
            model.save(self.get_models_last_filename_to_generate(process_name))
            print("OK")
        else:
            print("This type of process is not accepted.")

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
        Creates a directory and print an error if it is not possible
        :param path: e.g "/random_dir/the_new_dir/
        :return: nothing
        """
        try:
            os.mkdir(path)
        except OSError:
            print(f"Couldn't create the dir : {path}")
        else:
            print(f"Successfully created the dir : {path}")

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
        :return: (tuple1 : 2 training tensors of features and labels, tuple2 : 2 validation tensors of features and labels)
        """
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # Normalize the data
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        # Reshape the data for training
        x_train = x_train.reshape((50000, 32 * 32 * 3))
        x_test = x_test.reshape((10000, 32 * 32 * 3))
        return (x_train, y_train), (x_test, y_test)

    def load_model(self, model_filename=None):
        """
        Returns a tensorflow keras model from the filename
        :param model_filename
        :return: a tensorflow keras model
        """
        try:
            return tf.keras.models.load_model(model_filename)
        except FileNotFoundError:
            print("Error: Couldn't load the model. Check if the file exists.")
