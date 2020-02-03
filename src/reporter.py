from tensorflow.keras.callbacks import Callback


class Reporter(Callback):

    def __init__(self, x_train, y_train, batch_size, model_name, log_file_path, hp_log_title=None):
        super().__init__()
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.epoch_iter = 0
        self.model_name = model_name
        self.log_file_path = log_file_path
        self.hp_log_title = hp_log_title.replace("\n", "")

    def on_train_begin(self, logs=None):
        f = open(self.log_file_path, "a")
        f.write(f"{'=' * 5}{self.model_name}({self.hp_log_title}){'=' * 5}\n")
        f.close()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_iter += 1
        # (loss, acc) = self.model.evaluate(self.x_train, self.y_train, batch_size=self.batch_size)
        f = open(self.log_file_path, "a")
        f.write("ep {} - "
                "l: {} ; "
                "acc : {} ; "
                "vl : {} ; "
                "vacc : {}\n"
                .format(self.epoch_iter, logs['loss'], logs['sparse_categorical_accuracy'], logs['val_loss'],
                        logs['val_sparse_categorical_accuracy'])
                )
        f.close()

    def on_train_end(self, logs=None):
        self.epoch_iter = 0
