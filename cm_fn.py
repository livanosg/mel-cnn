import io
import itertools

import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
from config import CLASSES_DICT
matplotlib.use('cairo')


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('cividis'))
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3)
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max(initial=0) / 2.

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
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


class CMLog(TensorBoard):
    def __init__(self, log_dir, eval_data, update_freq):
        super().__init__(log_dir=log_dir, update_freq=update_freq, histogram_freq=0, write_graph=False,
                         write_images=False, profile_batch=0, embeddings_freq=0, embeddings_metadata=None)
        self.eval_data = eval_data

    def on_epoch_end(self, epoch, logs=None):
        test_pred_raw = self.model.predict(self.eval_data)
        test_pred = np.argmax(test_pred_raw, axis=1)
        labels = np.concatenate([np.argmax(label[1]['classes'], axis=1)
                                 for label in self.eval_data.as_numpy_iterator()])
        cm = np.asarray(tf.math.confusion_matrix(labels=labels, predictions=test_pred, num_classes=len(CLASSES_DICT)))
        figure = plot_confusion_matrix(cm, class_names=CLASSES_DICT.keys())
        cm_image = plot_to_image(figure)
        # Log the confusion matrix as an image summary.
        with tf.summary.create_file_writer(logdir=self.log_dir).as_default(step=epoch):
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
        super(CMLog, self).on_epoch_end(epoch=epoch)
