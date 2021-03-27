import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.python.data.experimental import AutoShardPolicy

from config import BATCH_SIZE, HWC
from dataset import MelData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# -----------------------------==================DISTRIBUTED SETUP =======================---------------------------- #
# # Fake SLURM ENVARS. Expected values for requesting 2 nodes (o1, o2)
# os.environ['SLURM_STEP_NUM_TASKS'] = '2'
# # len(SLURM_STEP_NODELIST) == len(SLURM_STEP_TASKS_PER_NODE)
# os.environ['SLURM_STEP_NODELIST'] = 'o[1,2]'  # example n[1-2],m5,o[3-4,6,7-9]')
# os.environ['SLURM_STEP_TASKS_PER_NODE'] = '1,1'  # example 3(x2),2,1
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
# Setup cluster
# https://stackoverflow.com/questions/66059593/multiworkermirroredstrategy-hangs-after-starting-grpc-server
# for i in range(int(os.environ['SLURM_STEP_NUM_TASKS'])):
#     os.environ['SLURM_PROCID'] = f'{i}'
#     slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(gpus_per_node=5, auto_set_gpu=False)
#     print(slurm_resolver.cluster_spec())
#     print(slurm_resolver.get_task_info())
#     print(slurm_resolver.environment)
# strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
#
# with strategy.scope():
# -----------------------------==================DISTRIBUTED SETUP =======================---------------------------- #
distr = tf.distribute.OneDeviceStrategy(device='/cpu')  # Testing distribution operations
with distr.scope():
    image_input = keras.Input(shape=HWC, batch_size=BATCH_SIZE, name='image')
    images_base_model = EfficientNetB0(include_top=False, input_tensor=image_input, weights='imagenet',
                                       drop_connect_rate=0.4)
    images_base_model.trainable = False
    reduce_base_model = keras.layers.Conv2D(100, kernel_size=1, padding='same')(images_base_model.output)
    flattened_basemodel_output = keras.layers.Flatten()(reduce_base_model)
    image_model_fcl_1 = keras.layers.Dense(int(flattened_basemodel_output.shape[-1] / 2), 'relu')(
        flattened_basemodel_output)
    image_model_fcl_2 = keras.layers.Dense(int(image_model_fcl_1.shape[-1] / 2), 'relu')(image_model_fcl_1)
    output_layer = keras.layers.Dense(5, activation='softmax', name='classes')(image_model_fcl_2)
    custom_model = tf.keras.Model(image_input, output_layer)
    custom_model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

# keras.utils.plot_model(custom_model, 'custom_model.png', rankdir='LR', show_layer_names=False)
train_data, eval_data = MelData()
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
train_data = train_data.with_options(options)
eval_data = eval_data.with_options(options)

tensor_call = tf.keras.callbacks.TensorBoard(update_freq=10, profile_batch=(2, 4))
custom_model.fit(train_data, validation_data=eval_data, epochs=2, callbacks=[tensor_call])

# test = np.asarray(tf.expand_dims(eval_data.as_numpy_iterator().next()[0]['image'][0], axis=0))
# activations = get_activations(model=custom_model, x=test)
# keract.display_activations(activations=activations, save=True, directory='test')
# keract.display_heatmaps(activations, test, save=True, directory='test')
