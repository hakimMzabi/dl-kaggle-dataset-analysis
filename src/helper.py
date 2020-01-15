import os
import matplotlib.pyplot as plt
from os import walk
import sqlite3
from datetime import datetime

'''
Helper class is here to help the developer debugging machine learning resources and variables and manage processes.
'''


class Helper:
    def __init__(self):
        self.accepted_processes = ["mlp", "convnet", "resnet", "rnn"]
        self.models_folder = "../responses/"

    def process_name_is_valid(self, process_name):
        return process_name.lower() in self.accepted_processes

    def get_models_last_filename(self, process_name):
        if self.process_name_is_valid(process_name) and os.path.isdir(self.models_folder):
            return self.models_folder + process_name.lower() + "_" + self.get_models_last_num(process_name) + ".h5"

    def get_models_last_num(self, process_name):
        if os.path.isdir(self.models_folder):
            max = 0
            for (dirpath, dirnames, filenames) in walk(self.models_folder):
                for filename in filenames:
                    if process_name.lower() in filename:
                        number = int(filename.split(process_name.lower() + "_")[1].split('.')[0])
                        if number > max:
                            max = number
            return str(max + 1)
        else:
            print(f"Error : The directory {self.models_folder} does not exists")
            exit()

    def save_model(self, model, process_name):
        model_filename = ""
        if self.process_name_is_valid(process_name):
            print(f"Generating save : {model_filename}...", end="")
            model.save(self.get_models_last_filename(process_name))
            print("OK")
        else:
            print("This type of process is not accepted.")

    @staticmethod
    def show_samples(x_train, y_train):
        for i in range(10):
            plt.imshow(x_train[i])
            print(y_train[i])
            plt.show()

    @staticmethod
    def create_dir(path):
        try:
            os.mkdir(path)
        except OSError:
            print(f"Couldn't create the dir : {path}")
        else:
            print(f"Successfully created the dir : {path}")

    # Show dataset shapes
    # Dataset must be equal to [x_train, y_train, x_test, y_test]
    @staticmethod
    def debug_dataset_shapes(dataset_name, dataset, terminate=False):
        print(f"[DEBUGGER] Debugging the {dataset_name} dataset :")
        print(f"[DEBUGGER]     x_train shape : {dataset[0].shape}")
        print(f"[DEBUGGER]     y_train shape : {dataset[1].shape}")
        print(f"[DEBUGGER]     x_test shape : {dataset[2].shape}")
        print(f"[DEBUGGER]     y_test shape : {dataset[3].shape}")
        if terminate:
            exit()
