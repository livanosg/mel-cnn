import io
import itertools

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.data.experimental import AutoShardPolicy
from tensorflow.python.keras.callbacks import TensorBoard

from config import CLASSES_DICT


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('cividis'))  # magma, inferno, plasma, viridis, cividis, twilight, twilight_shifted, turbo
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3)
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    buf = io.BytesIO()
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Use tf.image.decode_png to convert the PNG buffer to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class CMTensorboard(TensorBoard):
    def __init__(self, eval_data, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
        options.experimental_threading.max_intra_op_parallelism = 1
        self.eval_data = eval_data.with_options(options=options)  #.repeat(1)

    def on_epoch_end(self, epoch, logs=None):
        test_pred_raw = self.model.predict(self.eval_data)
        test_pred = np.argmax(test_pred_raw, axis=1)
        labels = np.concatenate([np.argmax(label['classes'], axis=1) for label in
                                 self.eval_data.map(lambda x, y: y).as_numpy_iterator()])
        cm = np.asarray(tf.math.confusion_matrix(labels=labels, predictions=test_pred, num_classes=len(CLASSES_DICT)))
        figure = plot_confusion_matrix(cm, class_names=CLASSES_DICT.keys())
        cm_image = plot_to_image(figure)
        file_writer = tf.summary.create_file_writer(self.log_dir + '/cm')
        # Log the confusion matrix as an image summary.
        with file_writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
        super().on_epoch_end(epoch, logs)
