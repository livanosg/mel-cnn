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
import os

from settings import MODELS_DIR

# load_model = '453432135'
load_model = 'nev_mel/clinic/130422070209'

if os.path.split(load_model)[0]:
    load_folder = load_model
else:
    load_folder = os.path.join('task', 'image_type', load_model)

if load_model:
    load_path = os.path.join(MODELS_DIR, load_folder)
