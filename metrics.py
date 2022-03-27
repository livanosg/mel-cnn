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
    tp = np.sum(y_pred * y_true)
    tn = np.sum((1-y_pred) * (1 - y_true))
    fp = np.sum(y_pred * (1 - y_true))
    fn = np.sum((1 - y_pred) * y_true)
    p = np.sum(y_true)
    pp = np.sum(y_pred)
    n = np.sum(1 - y_true)
    pn = np.sum(1 - y_pred)
    return tp, tn, fp, fn, p, pp, n, pn


def f_beta(beta, precision, recall):
    return (1 + np.power(beta, 2)) * ((precision * recall) / ((np.power(beta, 2) * precision) + recall))


def metrics(TP, TN, FP, FN, P, PP, N, PN):
    metrics_dict = {}
    metrics_dict['accuracy'] = np.round((TP + TN) / (P + N), 3)
    metrics_dict['sensitivity'] = np.round(TP / P, 3)
    metrics_dict['specificity'] = np.round(TN / N, 3)
    metrics_dict['precision'] = np.round(TP / PP, 3)
    metrics_dict['balanced_accuracy'] = np.round((metrics_dict['sensitivity'] + metrics_dict['specificity']) / 2, 3)
    metrics_dict['F1'] = np.round(f_beta(beta=1, precision=metrics_dict['precision'], recall=metrics_dict['sensitivity']), 3)
    metrics_dict['F2'] = np.round(f_beta(beta=2, precision=metrics_dict['precision'], recall=metrics_dict['sensitivity']), 3)
    return metrics_dict


def calc_metrics(args, model, dataset, dataset_type, dist_thresh=None, f1_thresh=None):
    save_dir = os.path.join(args['dir_dict']['trial'], dataset_type + '_{}'.format(args['image_type']))
    os.makedirs(save_dir, exist_ok=True)
    output = np.expand_dims(np.empty_like(model.output_shape), axis=0)

    if dataset_type != 'isic20_test':
        paths = np.empty_like(dataset.element_spec[0]['image_path'].shape)
        labels = np.expand_dims(np.empty_like(dataset.element_spec[1]['class'].shape), axis=0)
    else:
        paths = np.empty_like(dataset.element_spec['image_path'])

        np.r_()
    for x in dataset.as_numpy_iterator():
        if dataset_type != 'isic20_test':
            paths = np.concatenate((paths, x[0]['image_path']))
            output = np.concatenate((output, model.predict(x[0])))
            labels = np.concatenate((labels, x[1]['class']))
        else:
            paths = np.concatenate(paths, x['image_path'])
            output = np.concatenate(output, model.predict(x))

    paths, output, labels = paths[1:, ...], output[1:, ...].astype(np.float), labels[1:, ...].astype(np.int)
    if len(paths) == 0:
        print('empty dataset')
        return
    else:
        print('dataset size: ', len(paths))

    df_dict = {'image_name': paths}
    for i, class_name in enumerate(TASK_CLASSES[args['task']]):
        df_dict[f'{class_name}'] = np.round(output[..., i], 5)
        if dataset_type != 'isic20_test':
            df_dict[f'{class_name + "_true"}'] = labels[..., i]
    df = pd.DataFrame(df_dict)
    df['image_name'] = df['image_name'].apply(lambda path: path.decode('UTF-8').replace(args['dir_dict']['data_folder'], ''))
    if dataset_type == 'isic20_test':
        df = pd.DataFrame({'image_name': df_dict['image_name'], 'target': output[..., 1]})
    df.to_csv(path_or_buf=os.path.join(save_dir, '{}_{}_results.csv'.format(dataset_type, args['image_type'])), index=False)

    if dataset_type != 'isic20_test':
        # Micro average (averaging the total true positives, false negatives and false positives)
        # is only shown for multi-label or multi-class with a subset of classes,
        # because it corresponds to accuracy otherwise and would be the same for all metrics.
        # One-vs-one. Computes the average AUC of all possible pairwise combinations of classes.
        # Insensitive to class imbalance when `average == 'macro'`.
        if args['task'] != '5cls':
            fpr_lst, tpr_lst, roc_thresh = roc_curve(y_true=labels[..., 1], y_score=output[..., 1], pos_label=1)
            prec_lst, rec_lst, pr_thresh = precision_recall_curve(y_true=labels[..., 1], probas_pred=output[..., 1], pos_label=1)
            dist = np.sqrt(np.power(fpr_lst, 2) + np.power(1 - tpr_lst, 2))  # Distance from (0,1)
            f1_values = np.multiply(2., np.divide(np.multiply(prec_lst, rec_lst), np.add(prec_lst, rec_lst)))  # F1 score for each threshold
            if dist_thresh == None:
                dist_thresh = roc_thresh[np.argmin(dist)]  # Threshold with minimum distance
            if f1_thresh == None:
                f1_thresh = pr_thresh[np.argmax(f1_values)]  # Threshold with maximum F1 score
            for threshold in (0.5, dist_thresh, f1_thresh):
                y_pred_thrs = np.greater_equal(output, threshold).astype(np.int32)
                cm_img = cm_image(y_true=labels[..., 1], y_pred=y_pred_thrs[..., 1], class_names=args['class_names'])
                with open(os.path.join(save_dir, "cm_{}.png".format(str(round(threshold, 2)))), "wb") as f:
                    f.write(cm_img)

                with open(os.path.join(save_dir, "report_{}.txt".format(str(round(threshold, 2)))), "w") as f:
                    f.write(classification_report(y_true=labels, y_pred=y_pred_thrs, target_names=args['class_names'], digits=3, zero_division=0))
                    f.write("{} {} {}\n".format(' '.rjust(12), 'thresh_dist'.rjust(10), 'thresh_f1'.rjust(10)))
                    f.write('{} {} {}\n'.format(' '.rjust(12), str(dist_thresh).rjust(10), str(f1_thresh).rjust(10)))

                with open(os.path.join(save_dir, 'metrics_{}.csv'.format(str(round(threshold, 2)))), 'w') as f:
                    f.write('Class,Balanced Accuracy,Precision,Sensitivity (Recall),Specificity,Accuracy,AUC,F1,F2,Average Precision\n')
                    for _class in range(len(args['class_names'])):
                        AP = np.round(average_precision_score(y_true=labels[..., _class], y_score=output[..., _class]), 3)
                        ROC_AUC = np.round(roc_auc_score(y_true=labels[..., _class], y_score=output[..., _class]), 3)
                        cm_vals = conf_mat(y_pred=y_pred_thrs[..., _class], y_true=labels[..., _class])
                        m_dict = metrics(*cm_vals)
                        f.write(f'{args["class_names"][_class]},{m_dict["balanced_accuracy"]},{m_dict["precision"]},'
                                f'{m_dict["sensitivity"]},{m_dict["specificity"]},{m_dict["accuracy"]},'
                                f'{ROC_AUC},{m_dict["F1"]},{m_dict["F2"]},{AP}\n')

            plt.figure(1), plt.title('ROC curve'), plt.xlabel('False positive rate'), plt.ylabel('True positive rate'), plt.gca().set_aspect('equal', adjustable='box')
            plt.plot([0, fpr_lst[np.argmin(dist)]], [1, tpr_lst[np.argmin(dist)]])
            plt.plot(fpr_lst, tpr_lst, label=' '.join([args['class_names'][1], '(AUC= {:.3f})'.format(ROC_AUC)])), plt.plot([0, 1], [0, 1], 'k--')
            plt.legend(loc='best')
            plt.figure(1), plt.savefig(os.path.join(save_dir, 'roc_curve.png'))

            # plt.figure(2), plt.title('PR curve'), plt.xlabel('Recall'), plt.ylabel('Precision'), plt.gca().set_aspect('equal', adjustable='box')
            # plt.plot(rec_lst, prec_lst, label=' '.join([args['class_names'][1], '(AP= {:.3f})'.format(AP)]))
            # plt.legend(loc='best')
            # plt.figure(2), plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
            plt.close('all')
        if dataset_type == 'validation':
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
    return buf.getvalue()


def cm_image(y_true, y_pred, class_names: list):
    figure = plot_confusion_matrix(cm=confusion_matrix(y_true=y_true, y_pred=y_pred), class_names=class_names)
    return plot_to_image(figure)
