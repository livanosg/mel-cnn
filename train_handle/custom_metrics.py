import os
import io
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from data_handle.features_def import TASK_CLASSES
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score, roc_curve, \
    precision_recall_curve, classification_report

from settings import parser


def gmean(y_true, y_pred):
    y_pred_arg = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.float32)
    y_true_arg = tf.cast(tf.argmax(y_true, axis=-1), dtype=tf.float32)
    tp = tf.reduce_sum(y_true_arg * y_pred_arg)
    tn = tf.reduce_sum((1. - y_true_arg) * (1. - y_pred_arg))
    fp = tf.reduce_sum((1 - y_true_arg) * y_pred_arg)
    fn = tf.reduce_sum(y_true_arg * (1 - y_pred_arg))
    sensitivity = tf.math.divide_no_nan(tp, tp + fn)
    specificity = tf.math.divide_no_nan(tn, tn + fp)
    g_mean = tf.math.sqrt(sensitivity * specificity)
    return g_mean


def f_beta(beta, precision, recall):
    return (1 + np.power(beta, 2)) * ((precision * recall) / ((np.power(beta, 2) * precision) + recall))


def metrics(y_pred, y_true):
    tp = np.sum(y_pred * y_true)
    tn = np.sum((1 - y_pred) * (1 - y_true))
    p = np.sum(y_true)
    pp = np.sum(y_pred)
    n = np.sum(1 - y_true)
    # fp = np.sum(y_pred * (1 - y_true))
    # fn = np.sum((1 - y_pred) * y_true)
    # pn = np.sum(1 - y_pred)

    metrics_dict = {'accuracy': np.round((tp + tn) / (p + n), 3),
                    'sensitivity': np.round(tp / p, 3),
                    'specificity': np.round(tn / n, 3),
                    'precision': np.round(tp / pp, 3)}
    metrics_dict['balanced_accuracy'] = np.round((metrics_dict['sensitivity'] + metrics_dict['specificity']) / 2, 3)
    metrics_dict['F1'] = np.round(f_beta(beta=1, precision=metrics_dict['precision'],
                                         recall=metrics_dict['sensitivity']), 3)
    metrics_dict['F2'] = np.round(f_beta(beta=2, precision=metrics_dict['precision'],
                                         recall=metrics_dict['sensitivity']), 3)
    metrics_dict['gmean'] = np.round(np.sqrt(metrics_dict['sensitivity'] * metrics_dict['specificity']), 3)
    return metrics_dict


def calc_metrics(model, args, dirs, dataset, dataset_name, dist_thresh=None, f1_thresh=None):
    print(f"Calculate metrics for {dataset_name} {args['image_type']}...")
    save_dir = os.path.join(dirs['trial'], '_'.join([dataset_name, args['image_type']]))
    os.makedirs(save_dir, exist_ok=True)
    if dataset_name != 'isic20_test':
        df_dict = {'image_name': np.concatenate(list(dataset.map(lambda samples, labels: samples['image_path'])))}
        labels = np.concatenate(list(dataset.map(lambda samples, labels: labels['class'])))
    else:
        df_dict = {'image_name': np.concatenate(list(dataset.map(lambda samples: samples['image_path'])))}
    output = model.predict(dataset)

    for i, class_name in enumerate(TASK_CLASSES[args['task']]):
        df_dict[f"{class_name}"] = np.round(output[:, i], 5)
        if dataset_name != 'isic20_test':
            df_dict[f"{class_name + '_true'}"] = labels[:, i]

    df = pd.DataFrame(df_dict)
    df['image_name'] = df['image_name'].apply(lambda path: path.decode('UTF-8').replace(dirs['proc_img_folder'], ''))
    # path.decode('UTF-8').replace(dirs['proc_img_folder'], '')
    if dataset_name == 'isic20_test':
        df = pd.DataFrame({'image_name': df_dict['image_name'], 'target': output[:, 1]})
    df.to_csv(path_or_buf=os.path.join(save_dir, '{}_{}_results.csv'.format(dataset_name, args['image_type'])),
              index=False)

    if dataset_name != 'isic20_test':
        # Micro average (averaging the total true positives, false negatives and false positives)
        # is only shown for multi-label or multi-class with a subset of classes,
        # because it corresponds to accuracy otherwise and would be the same for all metrics.
        # One-vs-one. Computes the average AUC of all possible pairwise combinations of classes.
        # Insensitive to class imbalance when `average == 'macro'`.
        if args['task'] != '5cls':
            fpr_lst, tpr_lst, roc_thresh = roc_curve(y_true=np.argmax(labels, axis=-1),
                                                     y_score=output[:, 1], pos_label=1)
            prec_lst, rec_lst, pr_thresh = precision_recall_curve(y_true=np.argmax(labels, axis=-1),
                                                                  probas_pred=output[:, 1], pos_label=1)
            dist = np.sqrt(np.power(fpr_lst, 2) + np.power(1 - tpr_lst, 2))  # Distance from (0,1)
            # F1 score for each threshold
            with np.errstate(divide='ignore', invalid='ignore'):
                f1_values = np.multiply(2., np.multiply(prec_lst, rec_lst) / np.add(prec_lst, rec_lst))
            if dist_thresh is None:
                dist_thresh = roc_thresh[np.argmin(dist)]  # Threshold with minimum distance
            if f1_thresh is None:
                f1_thresh = pr_thresh[np.argmax(f1_values)]  # Threshold with maximum F1 score
            for threshold in (0.5,):  # dist_thresh, f1_thresh
                y_pred_thrs = np.greater_equal(output, threshold).astype(np.int32)
                cm_img = cm_image(y_true=np.argmax(labels, axis=-1), y_pred=y_pred_thrs[:, 1],
                                  class_names=TASK_CLASSES[args['task']])
                with open(os.path.join(save_dir, "cm_{}.png".format(str(round(threshold, 2)))), "wb") as f:
                    f.write(cm_img)

                with open(os.path.join(save_dir, "report_{}.txt".format(str(round(threshold, 2)))), "w") as f:
                    f.write(classification_report(y_true=labels, y_pred=y_pred_thrs,
                                                  target_names=TASK_CLASSES[args['task']], digits=3, zero_division=0))
                    f.write("{} {} {}\n".format(' '.rjust(12), 'thresh_dist'.rjust(10), 'thresh_f1'.rjust(10)))
                    f.write('{} {} {}\n'.format(' '.rjust(12), str(dist_thresh).rjust(10), str(f1_thresh).rjust(10)))

                with open(os.path.join(save_dir, 'metrics_{}.csv'.format(str(round(threshold, 2)))), 'w') as f:
                    f.write('Class,Balanced Accuracy,Precision,Sensitivity (Recall),Specificity,Accuracy,AUC,F1,F2,'
                            'G-Mean,Average Precision\n')
                    for _class in range(len(TASK_CLASSES[args['task']])):
                        AP = np.round(average_precision_score(y_true=labels[:, _class], y_score=output[:, _class]), 3)
                        ROC_AUC = np.round(roc_auc_score(y_true=labels[:, _class], y_score=output[:, _class]), 3)
                        m_dict = metrics(y_pred=y_pred_thrs[:, _class], y_true=labels[:, _class])
                        f.write(
                            f"{TASK_CLASSES[args['task']][_class]},{m_dict['balanced_accuracy']},{m_dict['precision']},"
                            f"{m_dict['sensitivity']},{m_dict['specificity']},{m_dict['accuracy']},"
                            f"{ROC_AUC},{m_dict['F1']},{m_dict['F2']},{m_dict['gmean']},{AP}\n")

            plt.figure(1)
            plt.title('ROC curve'), plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('False positive rate'), plt.ylabel('True positive rate')
            plt.plot(fpr_lst, tpr_lst, label=' '.join([TASK_CLASSES[args['task']][1], '(AUC= {:.3f})'.format(ROC_AUC)]))
            plt.plot([0, 1], [0, 1], 'k--'), plt.legend(loc='best')
            plt.figure(1), plt.savefig(os.path.join(save_dir, 'roc_curve.png'))

            plt.figure(2)
            plt.title('PR curve'), plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('Recall'), plt.ylabel('Precision')
            plt.plot(rec_lst, prec_lst, label=' '.join([TASK_CLASSES[args['task']][1], '(AP= {:.3f})'.format(AP)]))
            plt.legend(loc='best')
            plt.figure(2), plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
            plt.close('all')
        if dataset_name == 'validation':
            return dist_thresh, f1_thresh
        else:
            return None, None


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(5, 5))
    normalized_cm = cm / np.expand_dims(cm.sum(axis=1), axis=-1)
    # Plot the normalized confusion matrix.
    plt.imshow(normalized_cm, cmap=plt.cm.get_cmap("binary"))  # https://matplotlib.org/1.2.1/_images/show_colormaps.png
    plt.clim(0., 1.)
    plt.colorbar(shrink=0.7, aspect=20 * 0.7)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names, rotation=90)
    # Labels from confusion matrix values.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if normalized_cm[i, j] > 0.5 else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    # plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    return figure


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    return buf.getvalue()


def cm_image(y_true, y_pred, class_names: list):
    figure = plot_confusion_matrix(cm=confusion_matrix(y_true=y_true, y_pred=y_pred), class_names=class_names)
    return plot_to_image(figure)


# @tf.keras.utils.register_keras_serializable(package="Addons")
class GMean(tf.keras.metrics.Metric):
    # From tensorflow addons F-beta Score implementation
    r"""Computes Geometric Mean of sensitivity and specificity.


    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is None.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
        name: (Optional) String name of the metric instance.
        dtype: (Optional) Data type of the metric result.

    Returns:
        Geometric Mean: float.

    Raises:
        ValueError: If the `average` has values other than
        `[None, 'micro', 'macro', 'weighted']`.

        ValueError: If the `beta` value is less than or equal
        to 0.

    `average` parameter behavior:

        None: Scores for each class are returned.

        micro: True positivies, false positives and
            false negatives are computed globally.

        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.

        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.

    """

    def __init__(
            self,
            num_classes,
            average=None,
            threshold=None,
            name="gmean",
            dtype=None,
            **kwargs,
    ):
        super().__init__(name=name, dtype=dtype)

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(
                "Unknown average type. Acceptable values "
                "are: [None, 'micro', 'macro', 'weighted']"
            )

        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError("The value of threshold should be a python float")
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError("threshold should be between 0 and 1")

        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold
        self.axis = None
        self.init_shape = []

        if self.average != "micro":
            self.axis = 0
            self.init_shape = [self.num_classes]

        def _zero_wt_init(name):
            return self.add_weight(
                name, shape=self.init_shape, initializer="zeros", dtype=self.dtype
            )

        self.true_positives = _zero_wt_init("true_positives")
        self.true_negatives = _zero_wt_init("true_negatives")
        self.false_positives = _zero_wt_init("false_positives")
        self.false_negatives = _zero_wt_init("false_negatives")
        self.weights_intermediate = _zero_wt_init("weights_intermediate")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold

        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        def _weighted_sum(val, sample_weight):
            if sample_weight is not None:
                val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1))
            return tf.reduce_sum(val, axis=self.axis)

        self.true_positives.assign_add(_weighted_sum(y_pred * y_true, sample_weight))
        self.false_positives.assign_add(_weighted_sum(y_pred * (1 - y_true), sample_weight))
        self.false_negatives.assign_add(_weighted_sum((1 - y_pred) * y_true, sample_weight))
        self.weights_intermediate.assign_add(_weighted_sum(y_true, sample_weight))
        self.true_negatives.assign_add(_weighted_sum((1 - y_pred) * (1 - y_true), sample_weight))

    def result(self):
        sensitivity = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )
        specificity = tf.math.divide_no_nan(
            self.true_negatives, self.true_negatives + self.false_positives
        )
        g_mean = tf.math.sqrt(sensitivity * specificity)

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.weights_intermediate, tf.reduce_sum(self.weights_intermediate)
            )
            g_mean = tf.reduce_sum(g_mean * weights)

        elif self.average is not None:  # [micro, macro]
            g_mean = tf.reduce_mean(g_mean)

        return g_mean

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "threshold": self.threshold,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        reset_value = tf.zeros(self.init_shape, dtype=self.dtype)
        tf.keras.backend.batch_set_value([(v, reset_value) for v in self.variables])


class GeometricMean(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """

    def __init__(self, name='geometric_mean', **kwargs):
        super(GeometricMean, self).__init__(name=name, **kwargs)  # handles base args (e.g., dtype)
        self.args = vars(parser().parse_args())
        self.num_classes = len(TASK_CLASSES[self.args['task']])
        self.total_cm = self.add_weight("total", shape=(self.num_classes, self.num_classes), initializer="zeros")

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true, y_pred))
        return self.total_cm

    def result(self):
        return self.process_confusion_matrix()

    def confusion_matrix(self, y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_pred = tf.argmax(y_pred, 1)
        y_true = tf.argmax(y_true, 1)
        return tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)

    def process_confusion_matrix(self):
        """returns gmean"""
        cm = self.total_cm
        sensitivity = tf.math.divide_no_nan(cm[1, 1], cm[1, 1] + cm[1, 0])
        specificity = tf.math.divide_no_nan(cm[0, 0], cm[0, 0] + cm[0, 1])
        return tf.math.sqrt(sensitivity * specificity)  # gmean

    def fill_output(self, output):
        output['g_mean'] = self.result()
