import os

import cv2.cv2 as cv2
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
# from config import MAPPER
from losses import weighted_categorical_crossentropy, custom_loss
from metrics import metrics
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve, auc, det_curve
from matplotlib import pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# matplotlib.use('TkAgg')
model_file = "/home/livanosg/projects/mel-cnn/trials/150721041250-run-0000_ben_mal/models/best-model"
test_file = "/home/livanosg/projects/mel-cnn/trials/150721041250-run-0000_ben_mal/test_data.csv"
test_data = pd.read_csv(test_file)
results = []
arg_results = []
labels = []



def prediction_input(data_dict):
    return {"image": tf.image.per_image_standardization(tf.reshape(tf.image.resize_with_pad(cv2.imread(data_dict["image"]), target_height=100, target_width=100), shape=(-1, 100, 100, 3))),
            'image_type': tf.reshape(tf.keras.backend.one_hot(indices=int(data_dict["image_type"]), num_classes=2), shape=[-1, 2]),
            'sex': tf.reshape(tf.keras.backend.one_hot(indices=data_dict["sex"], num_classes=2), shape=[-1, 2]),
            'anatom_site_general': tf.reshape(tf.keras.backend.one_hot(indices=int(data_dict["anatom_site_general"]), num_classes=6), shape=[-1, 6]),
            'age_approx': tf.reshape(tf.keras.backend.one_hot(indices=int(data_dict["age_approx"]), num_classes=10), shape=[-1, 10])}


with tf.device('CPU:0'):
    model = tf.keras.models.load_model(model_file,
                                       custom_objects={
                                           'total_loss': custom_loss(weights=[0.5623483, 4.5097322]),
                                           "metrics": metrics})
    for i, row in test_data.iterrows():
        model_input = prediction_input(row)
        output = model.predict(model_input)
        results.append(np.asarray(tf.squeeze(output)[1]))
        arg_results.append(np.asarray(tf.squeeze(tf.argmax(output, -1))))
        labels.append(np.asarray(row["class"]))

y_true = np.asarray(labels)
arg_y_pred = np.asarray(arg_results)
y_pred = np.asarray(results)
print(f1_score(y_true=y_true, y_pred=arg_y_pred, average="weighted"))

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)
precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
det_fpr, det_fnr, det_thresholds = det_curve(y_true=y_true, y_score=y_pred)

auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

plt.figure(2)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(precision, recall, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('precision')
plt.ylabel('recall')
plt.title('PR curve')
plt.legend(loc='best')
plt.show()

plt.figure(3)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(det_fpr, det_fnr, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('DET curve')
plt.legend(loc='best')
plt.show()
