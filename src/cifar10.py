from src.helper import Helper

class Cifar10:
    def __init__(self, dim=1):
        self.helper = Helper()
        if dim == 1:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.helper.get_cifar10_prepared()
        elif dim == 3:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.helper.get_cifar10_prepared(dim=3)
        else:
            raise Exception("Cifar 10 couldn't be initialized with dims != 3 or != 1")
