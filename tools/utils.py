# import tensorflow as tf
# import numpy as np
# import keract
# from keract import get_activations
#
#
# def visuals(model, eval_data):
#     tf.keras.utils.plot_model(model, 'custom_model.png', rankdir='LR', show_layer_names=False)
#     test = np.asarray(tf.expand_dims(eval_data.as_numpy_iterator().next()[0]['image'][0], axis=0))
#     activations = get_activations(model=model, x=test)
#     keract.display_activations(activations=activations, save=True, directory='test')
#     keract.display_heatmaps(activations, test, save=True, directory='test')
import csv
import os
from config import DATA_DIR, MAIN_DIR

with open(os.path.join(DATA_DIR, 'mclass_clinic_test.csv')) as first_csv, open(os.path.join(MAIN_DIR, 'data_train.csv')) as second_csv:
    first_csv = csv.DictReader(first_csv)
    second_csv = csv.DictReader(second_csv)
    first_lst = []
    second_lst = []
    count = 0
    for row in first_csv:
        first_lst.append(row['image'])

    for row in second_csv:
        second_lst.append(row['image'].split('/')[-1])

    total = set(first_lst).intersection(set(second_lst))
    print(len(total))
    print(total)


    # print(total) 9 val

    # print("".join([word+"\n" for word in set(file1.read().split()) & set(file2.read().split())]))