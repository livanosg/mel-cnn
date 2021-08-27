import os
import io
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, use as plt_use
from config import TASK_CLASSES
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, classification_report
plt_use('Agg')


def calc_metrics(args, model, dataset, dataset_type):
    save_dir = os.path.join(args['dir_dict']['trial'], dataset_type)
    os.makedirs(save_dir, exist_ok=True)
    if dataset_type == 'isic20_test':
        results = []
        paths = []
        for x in dataset.as_numpy_iterator():
            y_prob = model.predict(x[0])
            if args['task'] == 'ben_mal':
                results.append(np.vstack(y_prob[..., 1]))
            else:  # Test 5cls performance in Benign-Malignant Task
                #  3: Non-Melanocytic Carcinoma | 4: Melanoma
                malignant = np.sum(y_prob[..., 3:], axis=-1)
                results.append(np.vstack(malignant))
            paths.append(np.vstack(x[1]))
        results = np.vstack(results).reshape((-1))
        paths = np.vstack(paths).reshape((-1))
        df = pd.DataFrame({'image_name': paths, 'target': results})
        df['image_name'] = df['image_name'].apply(
            lambda path: os.path.splitext(os.path.basename(path.decode('UTF-8')))[0])
        # noinspection PyTypeChecker
        df.to_csv(path_or_buf=os.path.join(save_dir, 'results.csv'), index=False)
    else:
        y_prob = model.predict(dataset)
        dataset_to_numpy = np.asarray([dt[1]['class'] for dt in dataset.as_numpy_iterator()])
        one_hot_labels = np.concatenate(dataset_to_numpy)
        y_true = np.argmax(one_hot_labels, axis=-1)
        y_pred = np.argmax(y_prob, axis=-1)
        cm_img = cm_image(y_true=y_true, y_pred=y_pred, class_names=args['class_names'])
        with open(os.path.join(save_dir, "cm.png"), "wb") as f:
            f.write(cm_img)
        with open(os.path.join(save_dir, "report.txt"), "w") as f:
            f.write(classification_report(y_true=y_true, y_pred=y_pred, target_names=args['class_names'], digits=3, zero_division=0))
            # In binary classification,
            # recall of the positive class is also known as "sensitivity";
            # recall of the negative class is also known as "specificity".
            # Micro average (averaging the total true positives, false negatives and false positives)
            # is only shown for multi-label or multi-class with a subset of classes,
            # because it corresponds to accuracy otherwise and would be the same for all metrics.
            # One-vs-one. Computes the average AUC of all possible pairwise combinations of classes. Insensitive to class imbalance when `average == 'macro'`.

        auc_macro_ovr = np.round(roc_auc_score(y_true=y_true, y_score=y_prob, multi_class='ovr', average='macro'), 3)
        auc_macro_ovo = np.round(roc_auc_score(y_true=y_true, y_score=y_prob, multi_class='ovo', average='macro'), 3)
        with open(os.path.join(save_dir, "report.txt"), "a") as f:
            col_1 = len(max(TASK_CLASSES[args['task']] + ['OvR_macro'], key=len))
            f.write("{} {}\n".format(''.rjust(max(col_1, 12), ' '), 'AUC'.rjust(10)))
            f.write("{} {}\n".format('{}'.format('OvR_macro').rjust(max(col_1, 12), ' '), '{}'.format(auc_macro_ovr).rjust(10)))
            f.write("{} {}\n".format('{}'.format('OvO_macro').rjust(max(col_1, 12), ' '), '{}'.format(auc_macro_ovo).rjust(10)))

        for _class in range(args['num_classes']):
            if args['num_classes'] == 2 and _class == 0:
                pass
            else:
                class_AP = np.round(average_precision_score(y_true=one_hot_labels[..., _class], y_score=y_prob[..., _class]), 3)
                class_roc_auc = np.round(roc_auc_score(y_true=one_hot_labels[..., _class], y_score=y_prob[..., _class]), 3)
                fpr_roc_curve, tpr_roc_curve, thresholds_roc = roc_curve(y_true=y_true, y_score=y_prob[..., _class], pos_label=_class)
                precision_curve, recall_curve, thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_prob[..., _class], pos_label=_class)
                with open(os.path.join(save_dir, "report.txt"), "a") as f:
                    f.write('{} {}\n'.format(TASK_CLASSES[args['task']][_class].rjust(max(col_1, 12), ' '), str(class_roc_auc).rjust(10)))
                plt.figure(1), plt.title('ROC curve'), plt.xlabel('False positive rate'), plt.ylabel('True positive rate')
                plt.plot(fpr_roc_curve, tpr_roc_curve, label=' '.join([args['class_names'][_class], '(AUC= {:.3f})'.format(class_roc_auc)])), plt.plot([0, 1], [0, 1], 'k--')
                plt.legend(loc='best')
                plt.figure(2), plt.title('PR curve'), plt.xlabel('Recall'), plt.ylabel('Precision'),
                plt.plot(recall_curve, precision_curve, label=' '.join([args['class_names'][_class], '(AP= {:.3f})'.format(class_AP)]))
                plt.legend(loc='best')
        plt.figure(1), plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
        plt.figure(2), plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
        plt.close('all')


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
    plt.colorbar(shrink=0.7, aspect=20*0.7)
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
    # Convert PNG buffer to TF image
    return buf.getvalue()


def cm_image(y_true, y_pred, class_names: list):
    cm = np.asarray(confusion_matrix(y_true=y_true, y_pred=y_pred))
    figure = plot_confusion_matrix(cm, class_names=class_names)
    return plot_to_image(figure)
