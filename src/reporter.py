import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class Reporter(Callback):

    def __init__(self, x_train, y_train, batch_size, model_name, log_file_path):
        super().__init__()
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.epoch_iter = 0
        self.model_name = model_name
        self.log_file_path = log_file_path

    def on_train_begin(self, logs=None):
        f = open(self.log_file_path, "a")
        f.write(f"{'=' * 5}{self.model_name}{'=' * 5}\n")
        f.close()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_iter += 1
        (loss, acc) = self.model.evaluate(self.x_train, self.y_train, batch_size=self.batch_size)
        f = open(self.log_file_path, "a")
        f.write("Epoch {} - real loss : {} - real accuracy : {:5.2f}%\n".format(self.epoch_iter, loss, 100 * acc))
        f.close()

    def on_train_end(self, logs=None):
        self.epoch_iter = 0
