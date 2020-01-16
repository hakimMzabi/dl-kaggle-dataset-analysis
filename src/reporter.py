from tensorflow.keras.callbacks import Callback


class Reporter(Callback):

    def __init__(self, x_train, y_train, batch_size, model_name):
        super().__init__()
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.epoch_iter = 0
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_iter += 1
        (loss, acc) = self.model.evaluate(self.x_train, self.y_train, batch_size=self.batch_size)
        print(f"Model name : {self.model_name}")
        print(f"Real loss on train : {loss}")
        print(f"Real accuracy on train : {acc}")

    def on_train_end(self, logs=None):
        self.epoch_iter = 0
