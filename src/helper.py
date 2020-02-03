import os
import re
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
import glob
import multiprocessing

from enum import Enum
from pathlib import Path
from src.reporter import Reporter
from tensorflow.keras.datasets import cifar10


class Helper:
    """
    Helper class is here to help the developer debugging machine learning resources and variables and manage processes.
    """

    class Bcolors(Enum):
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    class Logt(Enum):
        LOSS = "loss"
        SPARSE_ACC = "sparse_categorical_accuracy"
        VAL_LOSS = "val_loss"
        VALL_SPARSE_ACC = "val_sparse_categorical_accuracy"
        DIGITS = 16

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
            else process_name + "_1"

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
        self.create_dir(self.models_responses_folder)
        max = 0
        for (dirpath, dirnames, filenames) in os.walk(self.models_responses_folder):
            for filename in filenames:
                if process_name.lower() in filename:
                    number = int(filename.split(process_name.lower() + "_")[1].split('.')[0])
                    if number > max:
                        max = number
        return max + 1

    def save_model(self, model, process_name) -> None:
        """
        Saves the model in the src/models/responses folder, automatically increments from the last model created
        :param model: tensorflow keras model
        :param process_name: mlp, resnet
        :return: None
        """
        model.save(self.get_models_last_filename_to_generate(process_name))

    @staticmethod
    def show_samples(x_train, y_train) -> None:
        """
        Show samples of an image dataset
        :param x_train: features
        :param y_train: labels
        :return: None
        """
        for i in range(10):
            plt.imshow(x_train[i])
            print(y_train[i])
            plt.show()

    @staticmethod
    def create_dir(path) -> None:
        """
        Creates a directory if it doesn't exist and print an error if it is not possible
        :param path: e.g "/random_dir/the_new_dir/
        :return: None
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
        """
        Convert a list to a string separated by semicolons
        :param list: e.g [0, 1, 2, 3] as a list object
        :return: [0;1;2;3] as a string
        """
        res = "["
        for el in list:
            res += str(el) + ";"
        res += "]"
        return res

    @staticmethod
    def score(acc, val_acc) -> float:
        p = min(acc, val_acc)
        g = max(acc, val_acc)
        return float(10 * p * (1 + (p / g)))

    @staticmethod
    def last_line(path) -> str:
        f = open(path)
        x = f.readlines()[-1]
        f.close()
        return x

    def get_mesures(self, el, path) -> tuple:
        dflt_res = "None"
        if ".log" in el:
            last_line = self.last_line(path + el)
            metrics = last_line.split(';')
            loss = metrics[0].split("-")[1].split(":")[1].strip()
            acc = metrics[1].split(':')[1].strip()
            val_loss = metrics[2].split(':')[1].strip()
            val_acc = metrics[3].split(':')[1].strip()
            return loss, acc, val_loss, val_acc
        return dflt_res, dflt_res, dflt_res, dflt_res

    def evaluate_models(self, n) -> dict:
        """
        Evaluates the current models
        :return: the n better models
        """
        path = self.src_path + "\\models\\logs\\"
        res = {}
        # model_eval = {}
        try:
            els = os.listdir(path)
            for k, v in enumerate(els):
                loss, acc, val_loss, val_acc = self.get_mesures(v, path)
                if loss != "None" and acc != "None" and val_loss != "None" and val_acc != "None":
                    res[v.strip(".log")] = self.score(float(str(acc)), float(str(val_acc)))
                    # model_eval[v.strip(".log")] = {"loss": loss, "acc": acc, "val_loss": val_loss, "val_acc": acc}
        except FileNotFoundError:
            print(f"Couldn't evaluate model since there is no logs in `{path}`")
        return {k: v for k, v in reversed(sorted(res.items(), key=lambda item: item[1])[:n])}

    @staticmethod
    def debug_dataset_shapes(dataset_name, dataset, terminate=False) -> None:
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
    def get_cifar10_prepared(dim=1) -> (tuple, tuple):
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
        if dim == 1:
            x_train = x_train.reshape((50000, 32 * 32 * 3))
            x_test = x_test.reshape((10000, 32 * 32 * 3))
        elif dim == 3:
            x_train = x_train.reshape((50000, 32, 32, 3))
            x_test = x_test.reshape((10000, 32, 32, 3))
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def load_model(model_filename=None) -> object:
        """
        Returns a tensorflow keras model from the filename
        :param model_filename
        :return: a tensorflow keras model instance
        """
        try:
            return tf.keras.models.load_model(model_filename)
        except FileNotFoundError:
            print("Error: Couldn't load the model. Check if the file exists.")

    def load_model(self, name, id):
        savepath = f"{self.src_path}\\models\\responses\\{name}_{id}.h5"
        return tf.keras.models.load_model(savepath)

    @staticmethod
    def create_file(path) -> None:
        """
        Create a file from path, directory must exists or file won't be created
        :param path: e.g folder1\\folder2\\file.txt
        :return: None
        """
        f = open(f"{path}", "w")
        f.close()

    def purge(self, model_name=None, ckpt=False):
        if ckpt:
            try:
                shutil.rmtree(f"{self.src_path}\\models\\checkpoints\\")
            except Exception:
                pass
        if model_name is not None:
            log_model = f"{self.src_path}\\models\\logs\\{model_name}.log"
            save_model = f"{self.src_path}\\models\\responses\\{model_name}.h5"
            tensorboard_dir = f"{self.src_path}\\models\\logs\\tensorboard\\fit\\"

            try:
                os.remove(log_model)
                os.remove(save_model)
                for f in os.listdir(tensorboard_dir):
                    if model_name in f:
                        shutil.rmtree(tensorboard_dir + f + "\\")
            except Exception:
                pass

    def fit(self, model, x_train, y_train, batch_size, epochs, validation_data, process_name,
            hp_log_title=None) -> None:
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
        :return: None
        """

        self.save_model(model, process_name)

        (x_test, y_test) = validation_data

        model_name = self.get_models_last_filename(process_name).split("\\")[-1].replace(".h5", "")

        log_file_dir = self.src_path + "\\models\\logs\\"
        checkpoint_file_dir = self.src_path + "\\models\\checkpoints\\"
        tensorboard_log_dir = self.src_path + "\\models\\logs\\tensorboard\\fit\\"

        self.create_dir(log_file_dir)
        self.create_dir(checkpoint_file_dir)
        self.create_dir(tensorboard_log_dir)

        log_file_path = log_file_dir + model_name + ".log"
        checkpoint_file_path = checkpoint_file_dir + model_name + ".ckpt"
        tensorboard_log_current_dir = tensorboard_log_dir + model_name + "_" + datetime.datetime.now() \
            .strftime("%Y%m%d-%H%M%S") + "\\"

        self.create_dir(tensorboard_log_current_dir)

        self.create_file(log_file_path)
        self.create_file(checkpoint_file_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_file_path,
            save_weights_only=True,
            verbose=1
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_current_dir, histogram_freq=1)

        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=[
                Reporter(x_train, y_train, batch_size, model_name, log_file_path, hp_log_title=hp_log_title),
                cp_callback,
                tensorboard_callback
            ]
        )
