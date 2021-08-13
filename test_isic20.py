import os
import numpy as np
import pandas as pd
import tensorflow as tf
from data_pipe import MelData


def test_isic20(args):
    args['batch_size'] = 256
    dataset = MelData(args=args)
    model = tf.keras.models.load_model(args["dir_dict"]["save_path"])
    test_data = dataset.get_dataset(mode='isic20_test')
    results = []
    paths = []
    for x in test_data.as_numpy_iterator():
        y_prob = model.predict(x[0])
        results.append(np.vstack(y_prob[..., 1]))
        paths.append(np.vstack(x[1]))
    results = np.vstack(results).reshape((-1))
    paths = np.vstack(paths).reshape((-1))
    df = pd.DataFrame({'image_name': paths, 'target': results})
    df.loc[:, 'image_name'].apply(lambda image_path: os.path.splitext(os.path.basename(image_path))[0])
    # noinspection PyTypeChecker
    df.to_csv(path_or_buf='results.csv', index=False)
