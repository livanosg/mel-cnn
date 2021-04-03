import json
import math
import os

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.python.data.experimental import AutoShardPolicy

from clr_callback import CyclicLR
from confusion_matrix_plot import CMTensorboard
from dataset import MelData, get_class_weights
from hyperparameters import RELU_A_LST, HWC_RNG, MODEL_LST, OPTIMIZER_LST, LR_LST, DROPOUT_LST, metrics, \
    BATCH_SIZE_RANGE
from log_lr_callback import LRTensorBoard

# -----------------------------==================DISTRIBUTED SETUP =======================---------------------------- #

# https://stackoverflow.com/questions/66059593/multiworkermirroredstrategy-hangs-after-starting-grpc-server]

# os.system("unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY")

# -----------------------------==================DISTRIBUTED SETUP =======================---------------------------- #
from losses import weighted_categorical_crossentropy


def set_tf_config(resolver, environment=None):
    """Set the TF_CONFIG env variable from the given cluster resolver"""
    cfg = {
        'cluster': resolver.cluster_spec().as_dict(),
        'task': {
            'type': resolver.get_task_info()[0],
            'index': resolver.get_task_info()[1],
        },
        'rpc_layer': resolver.rpc_layer,
    }
    if environment:
        cfg['environment'] = environment
    os.environ['TF_CONFIG'] = json.dumps(cfg)


def training(hparams, log_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    # slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
    # set_tf_config(slurm_resolver)
    # distr = tf.distribute.MultiWorkerMirroredStrategy()

    distr = tf.distribute.OneDeviceStrategy(device='/cpu')  # Testing distribution operations
    print(f'Number of replicas in sync: {distr.num_replicas_in_sync}')

    models = {'xception': (Xception, tf.keras.applications.xception.preprocess_input),
              'inception': (InceptionV3, tf.keras.applications.inception_v3.preprocess_input),
              'efficientnet0': (EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
              'efficientnet1': (EfficientNetB1, tf.keras.applications.efficientnet.preprocess_input)}

    def relu(alpha):
        return tf.keras.layers.LeakyReLU(alpha=alpha)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    options.experimental_threading.max_intra_op_parallelism = 1
    datasets = MelData(hparams=hparams)
    train_data, eval_data = datasets.get_datasets()
    train_data, eval_data = train_data.with_options(options), eval_data.with_options(options)
    weights = get_class_weights(datasets.train_data)

    with distr.scope():
        # -----------------------------================ Image part =================---------------------------------- #
        image_input = keras.Input(shape=(hparams[HWC_RNG], hparams[HWC_RNG], 3), name='image')
        base_model_preproc = models[hparams[MODEL_LST]][1](image_input)
        base_model = models[hparams[MODEL_LST]][0](include_top=False, input_tensor=base_model_preproc)
        base_model.trainable = False
        reduce_base = keras.layers.Conv2D(128, kernel_size=1, padding='same')(base_model.output)
        flat = keras.layers.Flatten()(reduce_base)
        img_fcl_1 = keras.layers.Dense(64, relu(alpha=hparams[RELU_A_LST]))(flat)
        img_fcl_2 = keras.layers.Dense(32, relu(alpha=hparams[RELU_A_LST]))(img_fcl_1)

        # -----------------------------================ Values part =================--------------------------------- #
        image_type_input = keras.Input(shape=(2,), name='image_type', dtype=tf.float32)
        sex_input = keras.Input(shape=(2,), name='sex', dtype=tf.float32)
        anatom_site_input = keras.Input(shape=(6,), name='anatom_site', dtype=tf.float32)
        age_input = keras.Input(shape=(1,), name='age', dtype=tf.float32)
        concat_inputs = keras.layers.Concatenate()([image_type_input, sex_input, anatom_site_input, age_input])
        concat_inputs = keras.layers.Dropout(rate=hparams[DROPOUT_LST])(concat_inputs)
        fc_1 = keras.layers.Dense(1024, relu(alpha=hparams[RELU_A_LST]), )(concat_inputs)
        fc_1 = keras.layers.Dropout(rate=hparams[DROPOUT_LST])(fc_1)
        fc_2 = keras.layers.Dense(512, relu(alpha=hparams[RELU_A_LST]))(fc_1)
        fc_2 = keras.layers.Dropout(rate=hparams[DROPOUT_LST])(fc_2)
        fc_3 = keras.layers.Dense(265, relu(alpha=hparams[RELU_A_LST]))(fc_2)
        fc_3 = keras.layers.Dropout(rate=hparams[DROPOUT_LST])(fc_3)
        fc_4 = keras.layers.Dense(128, relu(alpha=hparams[RELU_A_LST]))(fc_3)
        fc_4 = keras.layers.Dropout(rate=hparams[DROPOUT_LST])(fc_4)
        fc_5 = keras.layers.Dense(64, relu(alpha=hparams[RELU_A_LST]))(fc_4)
        fc_5 = keras.layers.Dropout(rate=hparams[DROPOUT_LST])(fc_5)
        fc_6 = keras.layers.Dense(32, relu(alpha=hparams[RELU_A_LST]))(fc_5)
        fc_6 = keras.layers.Dropout(rate=hparams[DROPOUT_LST])(fc_6)

        # -----------------------------================= Concat part =================---------------------------------#
        final_concat = keras.layers.Concatenate()([fc_6, img_fcl_2])
        fc_all = keras.layers.Dense(32, activation=relu(alpha=hparams[RELU_A_LST]))(final_concat)
        fc_all = keras.layers.Dropout(rate=hparams[DROPOUT_LST])(fc_all)
        output_layer = keras.layers.Dense(5, activation='softmax', name='classes')(fc_all)
        custom_model = tf.keras.Model([image_input, image_type_input, sex_input, anatom_site_input, age_input],
                                      [output_layer])

        custom_model.compile(hparams[OPTIMIZER_LST], loss=weighted_categorical_crossentropy(weights),
                             metrics=metrics())  # 'categorical_crossentropy'

    steps_per_epoch = math.ceil(datasets.train_len / hparams[BATCH_SIZE_RANGE])

    tensorboard_callback = TensorBoard(log_dir=log_dir, update_freq='epoch', profile_batch=(1, 5))
    model_ckpt_callback = ModelCheckpoint(filepath='models/' + log_dir.split('/')[-1] + '-{epoch:03d}', save_freq='epoch',
                                          monitor='val_accuracy', save_best_only=True)
    # steps_per_epoch < step_size | step_size 2-8 x steps_per_epoch
    clr_callback = CyclicLR(base_lr=hparams[LR_LST], max_lr=hparams[LR_LST] * 5,
                            step_size=steps_per_epoch * 2, mode='exp_range', gamma=0.999)
    lr_log_callback = LRTensorBoard(log_dir=log_dir, update_freq='epoch', profile_batch=0)
    hp_callback = hp.KerasCallback(log_dir, hparams)
    es_callback = EarlyStopping(monitor='val_accuracy', verbose=1, patience=10, mode='max')
    cm_callback = CMTensorboard(log_dir=log_dir, update_freq='epoch', eval_data=eval_data, profile_batch=0)
    custom_model.fit(train_data, validation_data=eval_data, epochs=30,
                     callbacks=[tensorboard_callback, model_ckpt_callback, clr_callback,
                                lr_log_callback, cm_callback, hp_callback, es_callback],
                     verbose=1)
    tf.keras.backend.clear_session()
