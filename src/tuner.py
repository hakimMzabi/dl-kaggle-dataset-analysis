"""
The Tuner class is here to automatically generate model generation scenarios present in the src/scenarios/ folder
The number of epochs for each scenarios is equal to 100.
"""
import os
import importlib

from src.cifar10 import Cifar10
from src.helper import Helper


class Tuner:
    def __init__(
            self,
            process_name=None,
            dropouts=None,
            dropout_values=None,
            optimizers=None,
            activation_functions=None,
            batch_sizes=None,
            filter_size=None,
            padding_values=None,
            kernel_sizes=None
    ):
        if process_name is not None:
            self.process_name = process_name
            self.dropouts = dropouts
            self.dropout_values = dropout_values
            self.optimizers = optimizers
            self.activation_functions = activation_functions
            self.batch_sizes = batch_sizes
            if process_name.lower() == "convnet":
                self.filter_size = filter_size
                self.padding_values = padding_values
                self.kernel_sizes = kernel_sizes
        self.helper = Helper()

    def mlp_write(self, scenario_file, dropout, optimizer, activation_function, batch_size):
        # TODO FINISH METHOD
        pass

    def convnet_write(self):
        # TODO FINISH METHOD and add args
        pass

    def write_in_scenario(self, scenario_file, dropout, optimizer, activation_function, batch_size):
        if self.process_name == "mlp":
            # TODO REPLACE  mlp_write(args..)
            if dropout == "DropoutDescending":
                scenario_file.write(f"{dropout}{self.helper.list_to_str_semic(self.dropout_values)},"
                                    f"{optimizer},{activation_function},{batch_size}\n")
            elif dropout == "DropoutConstant":
                for dropout_value in self.dropout_values:
                    scenario_file.write(
                        f"{dropout}[{(str(dropout_value) + ';') * len(self.dropout_values)}],"
                        f"{optimizer},{activation_function},{batch_size}\n")
            else:
                scenario_file.write(
                    f"{dropout},{optimizer},{activation_function},{batch_size}\n")
        elif self.process_name == "convnet":
            # TODO REPLACE  convnet_write(args..)
            pass


    def create_scenario(self, scenario_name):
        self.helper.create_dir(self.helper.src_path + "\\scenarios")
        self.helper.create_dir(self.helper.src_path + "\\scenarios\\" + self.process_name)
        scenario_file_path = self.helper.src_path + "\\scenarios\\" + self.process_name + "\\" + scenario_name + ".csv"
        try:
            scenario_file = open(scenario_file_path, "w")
            for dropout in self.dropouts:
                for optimizer in self.optimizers:
                    for activation_function in self.activation_functions:
                        for batch_size in self.batch_sizes:
                            self.write_in_scenario(
                                scenario_file,
                                dropout,
                                optimizer,
                                activation_function,
                                batch_size
                            )
            scenario_file.close()
        except ValueError:
            print(f"Error: Could not create a file \"{scenario_file_path}\".")
        else:
            print(f"Successfully created the \"{scenario_file_path}\" file.")

    def filter_dropout(self, dropout_str):
        if "[" in dropout_str:
            (dropout_type, dropout_values) = dropout_str.split("[")
            dropout_values = dropout_values.replace("]", "").split(";")[:-1]
            return dropout_type, [float(i) for i in dropout_values]
        return dropout_str, None

    def launch_scenario(
            self,
            process_name,
            scenario_name,
            x_train,
            y_train,
            x_test,
            y_test,
            epochs
    ):
        scenario_file_path = self.helper.src_path + "\\scenarios\\" + process_name + "\\" + scenario_name + ".csv"
        scenario_file = open(scenario_file_path, "r")
        process = importlib.import_module("src.models.processes." + process_name)
        for line in scenario_file:
            hp = list(map(str.strip, line.split(",")))
            (dropout, dropout_values) = self.filter_dropout(hp[0])
            optimizer = hp[1]
            activation_function = hp[2]
            batch_size = int(hp[3])
            model = process.create_model(
                optimizer=optimizer,
                dropout_values=dropout_values,
                activation=activation_function,
            )
            model.summary()
            self.helper.fit(
                model,
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                process_name=process_name
            )

        scenario_file.close()


def mlp_tuner():
    # Create tuner
    mlp_tuner = Tuner(
        "mlp",
        dropouts=["NoDropout", "DropoutDescending", "DropoutConstant"],
        dropout_values=[0.2, 0.1],
        optimizers=["SGD", "Adam", "Adamax"],
        activation_functions=["tanh", "relu", "sigmoid"],
        batch_sizes=[32, 64, 128, 256]
    )

    # Load dataset
    cifar10 = Cifar10(dim=3)
    # (x_train, y_train), (x_test, y_test) = mlp_tuner.helper.get_cifar10_prepared()

    mlp_tuner.create_scenario("scenario_1")

def convnet_tuner():
    # Create tuner
    convnet_tuner = Tuner(
        "convnet",
        dropouts=["NoDropout", "DropoutDescending", "DropoutConstant"],
        dropout_values=[0.5, 0.4, 0.3],
        activation_functions=["tanh", "relu", "sigmoid"],
        batch_sizes=[32, 64, 128, 256],
        filter_size=64,
        padding_values=["same", "valid"],
        kernel_sizes=[
            (3, 3),
            (4, 4),
            (5, 5)
        ]
    )

def mlp_scenario_launcher():
    cifar10 = Cifar10(dim=1)
    tuner = Tuner()
    tuner.launch_scenario(
        "mlp",
        "scenario_1",
        cifar10.x_train,
        cifar10.y_train,
        cifar10.x_test,
        cifar10.y_test,
        epochs=100
    )

if __name__ == "__main__":
    mlp_scenario_launcher()
