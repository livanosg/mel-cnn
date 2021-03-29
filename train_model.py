import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.python.data.experimental import AutoShardPolicy
from tensorboard.plugins.hparams import api as hp
from dataset import MelData

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# -----------------------------==================DISTRIBUTED SETUP =======================---------------------------- #
# # Fake SLURM ENVARS. Expected values for requesting 2 nodes (o1, o2)
# os.environ['SLURM_STEP_NUM_TASKS'] = '1'
# # len(SLURM_STEP_NODELIST) == len(SLURM_STEP_TASKS_PER_NODE)
# os.environ['SLURM_STEP_NODELIST'] = 'white-rabbit,white-rabbit'  # example n[1-2],m5,o[3-4,6,7-9]')
# os.environ['SLURM_STEP_TASKS_PER_NODE'] = '1,1'  # example 3(x2),2,1
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Setup cluster
# https://stackoverflow.com/questions/66059593/multiworkermirroredstrategy-hangs-after-starting-grpc-server]
# os.system("unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY")

# -----------------------------==================DISTRIBUTED SETUP =======================---------------------------- #
# slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
# distr = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)


# keras.utils.plot_model(custom_model, 'custom_model.png', rankdir='LR', show_layer_names=False)

# test = np.asarray(tf.expand_dims(eval_data.as_numpy_iterator().next()[0]['image'][0], axis=0))
# activations = get_activations(model=custom_model, x=test)
# keract.display_activations(activations=activations, save=True, directory='test')
# keract.display_heatmaps(activations, test, save=True, directory='test')
