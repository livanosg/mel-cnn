import os
import io
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc, classification_report, confusion_matrix
from matplotlib import pyplot as plt, use as plt_use
from config import CLASS_NAMES
plt_use('cairo')


def calc_metrics(args, model, dataset, dataset_type):
    save_dir = os.path.join(args['dir_dict']['trial'], dataset_type)
    os.makedirs(save_dir, exist_ok=True)
    y_prob = model.predict(dataset)
    if args['test']:
        results = []
        paths = []
        for x in dataset.as_numpy_iterator():
            y_prob = model.predict(x[0])
            results.append(np.vstack(y_prob[..., 1]))
            paths.append(np.vstack(x[1]))
        results = np.vstack(results).reshape((-1))
        paths = np.vstack(paths).reshape((-1))
        df = pd.DataFrame({'image_name': paths, 'target': results})
        df.loc[:, 'image_name'].apply(lambda image_path: os.path.splitext(os.path.basename(str(image_path, 'UTF-8')))[0])
        # noinspection PyTypeChecker
        df.to_csv(path_or_buf=os.path.join(save_dir, 'results.csv'), index=False)
    else:
        dataset_to_numpy = np.asarray(list(map(lambda dt: dt[1]['class'], dataset.as_numpy_iterator())), dtype=object)
        one_hot_labels = np.concatenate(dataset_to_numpy)
        y_true = np.argmax(one_hot_labels, axis=-1)
        y_pred = np.argmax(y_prob, axis=-1)

        confmat_image = cm_image(y_true=y_true, y_pred=y_pred, class_names=args['class_names'])
        with open(os.path.join(save_dir, "cm.png"), "wb") as f:
            f.write(confmat_image)
        with open(os.path.join(save_dir, "report.txt"), "w") as f:
            labels = list(range(len(args['class_names'])))
            f.write(classification_report(y_true=y_true, y_pred=y_pred,
                                          target_names=list([args['class_names'][i] for i in labels]), labels=labels,
                                          digits=3, zero_division=0))
            # In binary classification,
            # recall of the positive class is also known as "sensitivity";
            # recall of the negative class is also known as "specificity".
            # Micro average (averaging the total true positives, false negatives and false positives)
            # is only shown for multi-label or multi-class with a subset of classes,
            # because it corresponds to accuracy otherwise and would be the same for all metrics.

        for _class in range(args['num_classes']):
            if args['num_classes'] == 2 and _class == 0:
                pass
            else:
                fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_true=y_true, y_score=y_prob[..., _class], pos_label=_class)
                precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_prob[..., _class],
                                                                       pos_label=_class)
                class_auc = auc(fpr_roc, tpr_roc)
                with open(os.path.join(save_dir, "report.txt"), "a") as f:
                    if (args['num_classes'] == 2 and _class == 1) or (args['num_classes'] != 2 and _class == 0):
                        col_1 = len(max(CLASS_NAMES[args['mode']], key=len))
                        f.write(f"{''.rjust(col_1, ' ')} {'AUC'.rjust(10)}\n")
                    f.write(
                        f"{CLASS_NAMES[args['mode']][_class].rjust(col_1)} {str(np.round(class_auc, 3)).rjust(10)}\n")
                plt.figure(1)
                plt.plot([0, 1], [0, 1], "k--")
                plt.plot(fpr_roc, tpr_roc, label=f"{args['class_names'][_class]} (area = {class_auc:.3f})")
                plt.xlabel("False positive rate")
                plt.ylabel("True positive rate")
                plt.title(f"ROC curve {args['image_type']}-{dataset_type}")
                plt.legend(loc="best")

                plt.figure(2)
                plt.plot(recall, precision, label=f"{args['class_names'][_class]}")
                plt.xlabel("Precision")
                plt.ylabel("Recall")
                plt.title(f"PR curve {args['image_type']}-{dataset_type}")
                plt.legend(loc="best")

            plt.figure(1)
            plt.savefig(os.path.join(save_dir, "roc.jpg"))
            plt.figure(2)
            plt.savefig(os.path.join(save_dir, "pr.jpg"))
            plt.close("all")


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation='nearest',
               cmap=plt.cm.get_cmap("binary"))  # https://matplotlib.org/1.2.1/_images/show_colormaps.png
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3)
    # labels = np.around(cm.astype('float'), decimals=3)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
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
