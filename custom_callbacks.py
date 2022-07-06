import gc

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from custom_metrics import cm_image


class MemFix(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch: int, logs=None):
        gc.collect()
        K.clear_session()


class EnrTensorboard(tf.keras.callbacks.TensorBoard):
    def __init__(self, val_data, class_names, **kwargs):
        super().__init__(**kwargs)
        self.class_names = class_names
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["learning_rate"] = self.model.optimizer.lr
        with super()._val_writer.as_default():
            tf.summary.image("Confusion Matrix",
                             tf.expand_dims(
                                 tf.image.decode_png(cm_image(
                                     y_true=np.argmax(np.concatenate(list(self.val_data.map(lambda samples, labels: labels['class']))), axis=-1),
                                     y_pred=np.argmax(self.model.predict(self.val_data), axis=-1), class_names=self.class_names), channels=3), 0),
                             step=epoch)
        super().on_epoch_end(epoch=epoch, logs=logs)


class LaterCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, start_at, **kwargs):
        super().__init__(**kwargs)
        self.start_at = start_at

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 >= self.start_at:
            if self.start_at == epoch + 1:
                print('Epoch {} reached. Start saving'.format(self.start_at))
            super().on_epoch_end(epoch=epoch, logs=logs)


class LaterReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self, start_at, **kwargs):
        super().__init__(**kwargs)
        self.start_at = start_at

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 >= self.start_at:
            if self.start_at == epoch + 1:
                print('Epoch {} reached. Start checking lr'.format(self.start_at))
            super().on_epoch_end(epoch=epoch, logs=logs)
