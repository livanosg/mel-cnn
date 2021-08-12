import os
import numpy as np
import pandas as pd
import tensorflow as tf

from data_pipe import MelData
from metrics import metrics


def test_isic20(args):
    # os.environ['TF_DISABLE_RZ_CHECK'] = '1'
    args['batch_size'] = 256
    dataset = MelData(args=args)
    model = tf.keras.models.load_model(args["dir_dict"]["save_path"], custom_objects={'metrics': metrics})
    test_data = dataset.get_dataset(mode='isic20_test')
    results = []
    paths = []
    for x in test_data.as_numpy_iterator():
        y_prob = model.predict(x[0])
        results.append(np.vstack(y_prob[..., 1]))
        paths.append(np.vstack(x[1]))
    results = np.vstack(results).reshape((-1))
    dataset_to_numpy = np.vstack(paths).reshape((-1))
    print(results.shape)
    print(dataset_to_numpy.shape)
    df = pd.DataFrame({'image_name': dataset_to_numpy, 'target': results})
    df.loc[:, 'image_name'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df.to_csv(path_or_buf='results.csv', index=False)

