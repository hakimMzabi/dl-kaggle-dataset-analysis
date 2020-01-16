"""
The Tuner class is here to automatically generate model generation scenarios present in the src/scenarios/ folder
The number of epochs for each scenarios is equal to 100.
"""


class Tuner:
    def __init__(self):
        self.model_types = ["MLP", "ConvNet", "ResNet", "RNN"]
        self.dropouts = [None, "Descending", "Constant"]
        self.optimizers = ["SGD", "Adagrad", "Adadelta", "Adam", "Adamax"]
        self.activation_functions = ["tanh", "relu", "sigmoid", "softmax"]

    def create_scenarios(self, scenario_file_name):
        path = "./scenarios/" + scenario_file_name
        try:
            scenario_file = open(path, "w")
            for model_type in self.model_types:
                for dropout in self.dropouts:
                    for optimizer in self.optimizers:
                        for activation_function in self.activation_functions:
                            scenario_file.write(
                                f"{model_type},{dropout},{optimizer},{activation_function}\n")
            scenario_file.close()
        except ValueError:
            print(f"Could not create a file {path}.")
        else:
            print(f"Successfully created the {path} file.")


if __name__ == "__main__":
    tuner = Tuner()
    tuner.create_scenarios("scenarios_generation_3.csv")
