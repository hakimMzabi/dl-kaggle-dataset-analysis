import os

'''
Helper class is here to help the developer debugging machine learning resources and variables and manage processes.
'''


class Helper:
    def __init__(self):
        pass

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
