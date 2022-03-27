import os
import io
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, use as plt_use
from config import TASK_CLASSES
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, classification_report
plt_use('Agg')

def conf_mat(y_pred, y_true):
    TP = np.sum(y_pred * y_true)
    TN = np.sum((1-y_pred) * (1 - y_true))
    FP = np.sum(y_pred * (1 - y_true))
    FN = np.sum((1 - y_pred) * y_true)
    P = np.sum(y_true)
    N = np.sum(1 - y_true)
    PP = np.sum(y_pred)
    PN = np.sum(1 - y_pred)
    return TP, TN, FP, FN, P, PP, N, PN

def f_beta(beta, precision, recall):
    return (1 + np.power(beta, 2)) * ((precision * recall) / ((np.power(beta, 2) * precision) + recall))

def metrics(TP, TN, FP, FN, P, PP, N, PN):
    sensitivity = np.round(TP / P, 3)
    specificity = np.round(TN / N, 3)
    balanced_accuracy = np.round((sensitivity + specificity) / 2, 3)
    accuracy = np.round((TP + TN) / (P + N), 3)
    precision = np.round(TP / PP, 3)
    F1 = np.round(f_beta(beta=1, precision=precision, recall=sensitivity), 3)
    F2 = np.round(f_beta(beta=2, precision=precision, recall=sensitivity), 3)
    return sensitivity, specificity, balanced_accuracy, accuracy, precision, F1, F2


# noinspection PyTypeChecker
def calc_metrics(args, model, dataset, dataset_type, thresh_dist=None, thresh_f1=None):
    save_dir = os.path.join(args['dir_dict']['trial'], dataset_type + '_{}'.format(args['image_type']))
    os.makedirs(save_dir, exist_ok=True)
    output = []
    labels = []
    paths = []
    for x in dataset.as_numpy_iterator():
        if dataset_type == 'isic20_test':
            paths.append(x['image_path'])
            output.append(model.predict(x))
        else:
            paths.append(x[0]['image_path'])
            output.append(model.predict(x[0]))
            labels.append(x[1]['class'])

    if len(paths) == 0:
        print('empty dataset')
        return
    paths = np.concatenate(paths)
    output = np.concatenate(output)
    if dataset_type != 'isic20_test':
        labels = np.concatenate(labels)

    df_dict = {'image_name': paths}
    df_dict.update({class_name: output[..., i] for i, class_name in enumerate(TASK_CLASSES[args['task']])})
    if dataset_type != 'isic20_test':
        df_dict.update({class_name+'_true': labels[..., i] for i, class_name in enumerate(TASK_CLASSES[args['task']])})
    df = pd.DataFrame(df_dict)
    df['image_name'] = df['image_name'].apply(lambda path: path.decode('UTF-8').replace(args['dir_dict']['data_folder'], ''))
    if dataset_type == 'isic20_test':
        df = pd.DataFrame({'image_name': paths, 'target': output[..., 1]})
    df.to_csv(path_or_buf=os.path.join(save_dir, '{}_{}_results.csv'.format(dataset_type, args['image_type'])), index=False)

    if dataset_type != 'isic20_test':
        y_true = np.argmax(labels, axis=-1)
        y_pred = np.argmax(output, axis=-1)  # Threshold 0.5
        cm_img = cm_image(y_true=y_true, y_pred=y_pred, class_names=args['class_names'])
        with open(os.path.join(save_dir, "cm_normal.png"), "wb") as f:
            f.write(cm_img)
        with open(os.path.join(save_dir, "report_normal.txt"), "w") as f:
            f.write(classification_report(y_true=y_true, y_pred=y_pred, target_names=args['class_names'], digits=3, zero_division=0))


        # Micro average (averaging the total true positives, false negatives and false positives)
        # is only shown for multi-label or multi-class with a subset of classes,
        # because it corresponds to accuracy otherwise and would be the same for all metrics.
        # One-vs-one. Computes the average AUC of all possible pairwise combinations of classes.
        # Insensitive to class imbalance when `average == 'macro'`.
        if args['num_classes'] == 2:
            AP = np.round(average_precision_score(y_true=labels[..., 1], y_score=output[..., 1]), 3)
            ROC_AUC = np.round(roc_auc_score(y_true=labels[..., 1], y_score=output[..., 1]), 3)
            fpr_values, tpr_values, thresh_roc = roc_curve(y_true=labels[..., 1], y_score=output[..., 1], pos_label=1)
            prec_values, rec_values, thresh_pr = precision_recall_curve(y_true=labels[..., 1], probas_pred=output[..., 1], pos_label=1)
            dist = np.sqrt(np.power(fpr_values, 2) + np.power(1 - tpr_values, 2))
            indx_dist = np.argmin(dist)
            f1_values = np.multiply(2., np.divide(np.multiply(prec_values, rec_values), np.add(prec_values, rec_values)))
            indx_f1 = np.argmax(f1_values)

            TP, TN, FP, FN, P, PP, N, PN = conf_mat(y_pred=y_pred, y_true=y_true)
            sensitivity, specificity, balanced_accuracy, accuracy, precision, F1, F2 = metrics(TP, TN, FP, FN, P, PP, N, PN)
            with open(os.path.join(save_dir, "metrics_normal.csv"), "w") as f:
                f.write('Balanced Accuracy,Precision, Sensitivity (Recall),Specificity,Accuracy,AUC,F1,F2, Average Precision\n')
                f.write(f'{balanced_accuracy},{precision},{sensitivity},{specificity},{accuracy},{ROC_AUC},{F1},{F2},{AP}')

            if dataset_type != 'test':
                # Distance from (0,1)
                if thresh_dist == None:
                    thresh_dist = thresh_roc[indx_dist]
                if thresh_f1 == None:
                    # F1 score for each threshold
                    thresh_f1 = thresh_pr[indx_f1]
            for threshold in (thresh_dist, thresh_f1):
                y_pred2 = np.greater_equal(output[..., 1], threshold).astype(np.int32)
                cm_img2 = cm_image(y_true=y_true, y_pred=y_pred2, class_names=args['class_names'])

                TP, TN, FP, FN, P, PP, N, PN = conf_mat(y_pred=y_pred2, y_true=y_true)
                sensitivity, specificity, balanced_accuracy, accuracy, precision, F1, F2 = metrics(TP, TN, FP, FN, P, PP, N, PN)

                with open(os.path.join(save_dir, "cm_{}.png".format(str(round(threshold, 2)))), "wb") as f:
                    f.write(cm_img2)
                with open(os.path.join(save_dir, "report_{}.txt".format(str(round(threshold, 2)))), "w") as f:
                    f.write(classification_report(y_true=y_true, y_pred=y_pred2, target_names=args['class_names'], digits=3, zero_division=0))
                    f.write('{} {} {}\n'.format(TASK_CLASSES[args['task']][1].rjust(12), str(ROC_AUC).rjust(10), str(AP).rjust(10)))
                    f.write("{} {} {}\n".format(' '.rjust(12), 'thresh_dist'.rjust(10), 'thresh_f1'.rjust(10)))
                    f.write('{} {} {}\n'.format(' '.rjust(12), str(thresh_dist).rjust(10), str(thresh_f1).rjust(10)))

                with open(os.path.join(save_dir, "metrics_{}.csv".format(str(round(threshold, 2)))), "w") as f:
                    f.write('Balanced Accuracy,Precision, Sensitivity (Recall),Specificity,Accuracy,AUC,F1,F2, Average Precision\n')
                    f.write(f'{balanced_accuracy},{precision},{sensitivity},{specificity},{accuracy},{ROC_AUC},{F1},{F2},{AP}')

            plt.figure(1), plt.title('ROC curve'), plt.xlabel('False positive rate'), plt.ylabel('True positive rate'), plt.gca().set_aspect('equal', adjustable='box')
            plt.plot([0, fpr_values[np.argmin(dist)]], [1, tpr_values[np.argmin(dist)]])
            plt.plot(fpr_values, tpr_values, label=' '.join([args['class_names'][1], '(AUC= {:.3f})'.format(ROC_AUC)])), plt.plot([0, 1], [0, 1], 'k--')
            plt.legend(loc='best')
            plt.figure(2), plt.title('PR curve'), plt.xlabel('Recall'), plt.ylabel('Precision'), plt.gca().set_aspect('equal', adjustable='box')
            plt.plot(rec_values, prec_values, label=' '.join([args['class_names'][1], '(AP= {:.3f})'.format(AP)]))
            plt.legend(loc='best')
            plt.figure(1), plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
            plt.figure(2), plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
            plt.close('all')
            if dataset_type == 'validation':
                return thresh_dist, thresh_f1
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
