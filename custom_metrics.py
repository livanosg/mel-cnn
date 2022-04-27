import os
import io
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from features_def import TASK_CLASSES
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score, roc_curve, \
    precision_recall_curve, classification_report


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
    logx = np.log10(metrics_dict['sensitivity'] / (1 - metrics_dict['sensitivity']))
    logy = np.log10(metrics_dict['specificity'] / (1 - metrics_dict['specificity']))
    metrics_dict['dp'] = np.round((np.sqrt(3) / np.pi) * (logx + logy), 3)
    return metrics_dict


def calc_metrics(model, args, dirs, dataset, dataset_name, dist_thresh=None, f1_thresh=None):
    save_dir = os.path.join(dirs['trial'], '_'.join([dataset_name, args['image_type']]))
    os.makedirs(save_dir, exist_ok=True)
    output = np.expand_dims(np.empty_like(model.output_shape), axis=0)
    labels = None
    if dataset_name != 'isic20_test':
        paths = np.empty_like(dataset.element_spec[0]['image_path'].shape)
        labels = np.expand_dims(np.empty_like(dataset.element_spec[1]['class'].shape), axis=0)
    else:
        paths = np.empty_like(dataset.element_spec['image_path'])

    for x in dataset.as_numpy_iterator():
        if dataset_name != 'isic20_test':
            paths = np.concatenate((paths, x[0]['image_path']))
            output = np.concatenate((output, model.predict(x[0])))
            labels = np.concatenate((labels, x[1]['class']))
        else:
            paths = np.concatenate(paths, x['image_path'])
            output = np.concatenate(output, model.predict(x))

    if dataset_name != 'isic20_test':
        labels = labels[1:, ...].astype(np.int)

    paths, output = paths[1:, ...], output[1:, ...].astype(np.float)
    if len(paths) == 0:
        print('empty dataset')
        return
    else:
        print('dataset size: ', len(paths))

    df_dict = {'image_name': paths}
    for i, class_name in enumerate(TASK_CLASSES[args['task']]):
        df_dict[f"{class_name}"] = np.round(output[..., i], 5)
        if dataset_name != 'isic20_test':
            df_dict[f"{class_name + '_true'}"] = labels[..., i]
    df = pd.DataFrame(df_dict)
    df['image_name'] = df['image_name'].apply(lambda path: path.decode('UTF-8').replace(dirs['proc_img_folder'], ''))
    # path.decode('UTF-8').replace(dirs['proc_img_folder'], '')
    if dataset_name == 'isic20_test':
        df = pd.DataFrame({'image_name': df_dict['image_name'], 'target': output[..., 1]})
    df.to_csv(path_or_buf=os.path.join(save_dir, '{}_{}_results.csv'.format(dataset_name, args['image_type'])),
              index=False)

    if dataset_name != 'isic20_test':
        # Micro average (averaging the total true positives, false negatives and false positives)
        # is only shown for multi-label or multi-class with a subset of classes,
        # because it corresponds to accuracy otherwise and would be the same for all metrics.
        # One-vs-one. Computes the average AUC of all possible pairwise combinations of classes.
        # Insensitive to class imbalance when `average == 'macro'`.
        if args['task'] != '5cls':
            fpr_lst, tpr_lst, roc_thresh = roc_curve(y_true=labels[:, 1],
                                                     y_score=output[:, 1], pos_label=1)
            prec_lst, rec_lst, pr_thresh = precision_recall_curve(y_true=labels[:, 1],
                                                                  probas_pred=output[:, 1], pos_label=1)
            dist = np.sqrt(np.power(fpr_lst, 2) + np.power(1 - tpr_lst, 2))  # Distance from (0,1)
            # F1 score for each threshold
            f1_values = np.multiply(2., np.divide(np.multiply(prec_lst, rec_lst), np.add(prec_lst, rec_lst)))
            if dist_thresh is None:
                dist_thresh = roc_thresh[np.argmin(dist)]  # Threshold with minimum distance
            if f1_thresh is None:
                f1_thresh = pr_thresh[np.argmax(f1_values)]  # Threshold with maximum F1 score
            for threshold in (0.5, dist_thresh, f1_thresh):
                y_pred_thrs = np.greater_equal(output, threshold).astype(np.int32)
                cm_img = cm_image(y_true=labels[:, 1], y_pred=y_pred_thrs[:, 1], class_names=TASK_CLASSES[args['task']])
                with open(os.path.join(save_dir, "cm_{}.png".format(str(round(threshold, 2)))), "wb") as f:
                    f.write(cm_img)

                with open(os.path.join(save_dir, "report_{}.txt".format(str(round(threshold, 2)))), "w") as f:
                    f.write(classification_report(y_true=labels, y_pred=y_pred_thrs,
                                                  target_names=TASK_CLASSES[args['task']], digits=3, zero_division=0))
                    f.write("{} {} {}\n".format(' '.rjust(12), 'thresh_dist'.rjust(10), 'thresh_f1'.rjust(10)))
                    f.write('{} {} {}\n'.format(' '.rjust(12), str(dist_thresh).rjust(10), str(f1_thresh).rjust(10)))

                with open(os.path.join(save_dir, 'metrics_{}.csv'.format(str(round(threshold, 2)))), 'w') as f:
                    f.write('Class,Balanced Accuracy,Precision,Sensitivity (Recall),Specificity,Accuracy,AUC,F1,F2,'
                            'G-Mean,Average Precision,Discriminant Power\n')
                    for _class in range(len(TASK_CLASSES[args['task']])):
                        AP = np.round(average_precision_score(y_true=labels[:, _class], y_score=output[:, _class]), 3)
                        ROC_AUC = np.round(roc_auc_score(y_true=labels[:, _class], y_score=output[:, _class]), 3)
                        m_dict = metrics(y_pred=y_pred_thrs[:, _class], y_true=labels[:, _class])
                        f.write(
                            f"{TASK_CLASSES[args['task']][_class]},{m_dict['balanced_accuracy']},{m_dict['precision']},"
                            f"{m_dict['sensitivity']},{m_dict['specificity']},{m_dict['accuracy']},"
                            f"{ROC_AUC},{m_dict['F1']},{m_dict['F2']},{m_dict['gmean']},{AP},{m_dict['dp']}\n")

            plt.figure(1)
            plt.title('ROC curve'), plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('False positive rate'), plt.ylabel('True positive rate')
            # plt.plot([0, fpr_lst[np.argmin(dist)]], [1, tpr_lst[np.argmin(dist)]])
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

    figure = plt.figure(figsize=(4.5, 4.5))
    normalized_cm = cm / np.expand_dims(cm.sum(axis=1), axis=-1)
    # Plot the normalized confusion matrix.
    plt.imshow(normalized_cm, interpolation='nearest',
               cmap=plt.cm.get_cmap("binary"))  # https://matplotlib.org/1.2.1/_images/show_colormaps.png
    plt.clim(0., 1.)
    plt.colorbar(shrink=0.7, aspect=20 * 0.7)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names, rotation=90)
    # Labels from confusion matrix values.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if normalized_cm[i, j] > 0.5 else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
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


class GeometricMean(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """

    def __init__(self, num_classes, **kwargs):
        super(GeometricMean, self).__init__(name='geometric_mean',
                                            **kwargs)  # handles base args (e.g., dtype)
        self.num_classes = num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes, num_classes), initializer="zeros")

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
        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)
        return cm

    def process_confusion_matrix(self):
        """returns gmean"""
        cm = self.total_cm
        # diag_part = tf.linalg.diag_part(cm)
        # precision = diag_part / (tf.reduce_sum(cm, 0) + tf.constant(1e-15))
        # recall = diag_part / (tf.reduce_sum(cm, 1) + tf.constant(1e-15))
        # f1 = 2 * precision * recall / (precision + recall + tf.constant(1e-15))
        sensitivity = tf.math.divide_no_nan(cm[1, 1], cm[1, 1] + cm[1, 0])
        specificity = tf.math.divide_no_nan(cm[0, 0], cm[0, 0] + cm[0, 1])
        g_mean = tf.math.sqrt(sensitivity * specificity)
        return g_mean

    def fill_output(self, output):
        results = self.result()
        output['g_mean'] = results

