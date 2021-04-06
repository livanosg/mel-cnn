import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams.api import KerasCallback
from tensorflow.python.data.experimental import AutoShardPolicy
from tensorflow.python.keras.callbacks import TensorBoard

from config import CLASSES_DICT
from confusion_matrix_plot import plot_confusion_matrix, plot_to_image


class AllTensorBoard(KerasCallback, TensorBoard):
    def __init__(self, log_dir, hparams, eval_data, update_freq, profile_batch):
        KerasCallback.__init__(self, writer=log_dir, hparams=hparams)
        TensorBoard.__init__(self, log_dir=log_dir, update_freq=update_freq, profile_batch=profile_batch)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
        options.experimental_threading.max_intra_op_parallelism = 1
        self.eval_data = eval_data.with_options(options=options)

    def on_train_begin(self, logs=None):
        if 'lr' not in logs.keys():
            print('lr not in keys')
            logs.update({'lr': self.model.optimizer.lr})
        TensorBoard.on_train_begin(self, logs=logs)
        KerasCallback.on_train_begin(self, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if 'lr' not in logs.keys():
            print('lr not in keys')
            logs.update({'lr': self.model.optimizer.lr})
        test_pred_raw = self.model.predict(self.eval_data)
        test_pred = np.argmax(test_pred_raw, axis=1)
        labels = np.concatenate([np.argmax(label['classes'], axis=1) for label in
                                 self.eval_data.map(lambda x, y: y).as_numpy_iterator()])
        cm = np.asarray(tf.math.confusion_matrix(labels=labels, predictions=test_pred, num_classes=len(CLASSES_DICT)))
        figure = plot_confusion_matrix(cm, class_names=CLASSES_DICT.keys())
        cm_image = plot_to_image(figure)
        with self._val_writer.as_default():
            # Log the confusion matrix as an image summary.
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
        self._val_writer.flush()
        TensorBoard.on_epoch_end(self, epoch=epoch, logs=logs)
