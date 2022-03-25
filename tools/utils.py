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

with open('/home/giorgos/projects/mel-cnn/isic18_val_test.csv') as test_id, open('/home/giorgos/projects/mel-cnn/isic20_test.csv') as test_total:
    train_csv = csv.DictReader(test_id)
    test_csv = csv.DictReader(test_total)
    train_lst = []
    test_lst = []
    count = 0
    for row in train_csv:
        train_lst.append(row['image'])

    for row in test_csv:
        test_lst.append(row['image'])

    total = set(train_lst).intersection(set(test_lst))
    print(len(total))
    print(total)


    # print(total)

    # print("".join([word+"\n" for word in set(file1.read().split()) & set(file2.read().split())]))