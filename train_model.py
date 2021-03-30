import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.data.experimental import AutoShardPolicy
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback, EarlyStopping
from clr_callback import CyclicLR
from confusion_matrix_plot import CMTensorboard
from dataset import MelData
from hyperparameters import ACTIVATION_OPTIONS, HWC_RANGE, HP_MODELS, HP_OPTIMIZER, metrics, LEARNING_RATE_RANGE, \
    DROPOUT_RANGE
from log_lr_callback import LRTensorBoard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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


def training(hparams, log_dir):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
    # distr = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
    distr = tf.distribute.OneDeviceStrategy(device='/cpu')  # Testing distribution operations
    models = {'xception': (Xception, tf.keras.applications.xception.preprocess_input),
              'inception': (InceptionV3, tf.keras.applications.inception_v3.preprocess_input),
              'efficientnet0': (EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
              'efficientnet1': (EfficientNetB1, tf.keras.applications.efficientnet.preprocess_input)}

    def relu(alpha):
        return tf.keras.layers.LeakyReLU(alpha=alpha)

    with distr.scope():
        # -----------------------------================ Image part =================----------------------------------- #
        image_input = keras.Input(shape=(hparams[HWC_RANGE], hparams[HWC_RANGE], 3), name='image')
        base_model_preproc = models[hparams[HP_MODELS]][1](image_input)
        base_model = models[hparams[HP_MODELS]][0](include_top=False, input_tensor=base_model_preproc)
        base_model.trainable = False
        reduce_base = keras.layers.Conv2D(128, kernel_size=1, padding='same')(base_model.output)
        flat = keras.layers.Flatten()(reduce_base)
        img_fcl_1 = keras.layers.Dense(64, relu(alpha=hparams[ACTIVATION_OPTIONS]))(flat)
        img_fcl_2 = keras.layers.Dense(32, relu(alpha=hparams[ACTIVATION_OPTIONS]))(img_fcl_1)

        # -----------------------------================ Values part =================--------------------------------- #
        image_type_input = keras.Input(shape=(2,), name='image_type', dtype=tf.float32)
        sex_input = keras.Input(shape=(3,), name='sex', dtype=tf.float32)
        anatom_site_input = keras.Input(shape=(8,), name='anatom_site', dtype=tf.float32)
        age_input = keras.Input(shape=(1,), name='age', dtype=tf.float32)
        concat_inputs = keras.layers.Concatenate()([image_type_input, sex_input, anatom_site_input, age_input])
        concat_inputs = keras.layers.Dropout(rate=hparams[DROPOUT_RANGE])(concat_inputs)
        fc_1 = keras.layers.Dense(1024, relu(alpha=hparams[ACTIVATION_OPTIONS]),)(concat_inputs)
        fc_1 = keras.layers.Dropout(rate=hparams[DROPOUT_RANGE])(fc_1)
        fc_2 = keras.layers.Dense(512, relu(alpha=hparams[ACTIVATION_OPTIONS]))(fc_1)
        fc_2 = keras.layers.Dropout(rate=hparams[DROPOUT_RANGE])(fc_2)
        fc_3 = keras.layers.Dense(265, relu(alpha=hparams[ACTIVATION_OPTIONS]))(fc_2)
        fc_3 = keras.layers.Dropout(rate=hparams[DROPOUT_RANGE])(fc_3)
        fc_4 = keras.layers.Dense(128, relu(alpha=hparams[ACTIVATION_OPTIONS]))(fc_3)
        fc_4 = keras.layers.Dropout(rate=hparams[DROPOUT_RANGE])(fc_4)
        fc_5 = keras.layers.Dense(64, relu(alpha=hparams[ACTIVATION_OPTIONS]))(fc_4)
        fc_5 = keras.layers.Dropout(rate=hparams[DROPOUT_RANGE])(fc_5)
        fc_6 = keras.layers.Dense(32, relu(alpha=hparams[ACTIVATION_OPTIONS]))(fc_5)
        fc_6 = keras.layers.Dropout(rate=hparams[DROPOUT_RANGE])(fc_6)

        # -----------------------------================= Concat part =================---------------------------------#
        final_concat = keras.layers.Concatenate()([fc_6, img_fcl_2])
        fc_all = keras.layers.Dense(32, activation=relu(alpha=hparams[ACTIVATION_OPTIONS]))(final_concat)
        fc_all = keras.layers.Dropout(rate=hparams[DROPOUT_RANGE])(fc_all)
        output_layer = keras.layers.Dense(5, activation='softmax', name='classes')(fc_all)
        custom_model = tf.keras.Model([image_input, image_type_input, sex_input, anatom_site_input, age_input],
                                      [output_layer])
        custom_model.compile(hparams[HP_OPTIMIZER], loss='categorical_crossentropy', metrics=metrics())

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    options.experimental_threading.max_intra_op_parallelism = 1

    train_data, eval_data = MelData(hparams=hparams)
    train_data, eval_data = train_data.with_options(options), eval_data.with_options(options)

    tensorboard_callback = TensorBoard(log_dir=log_dir, update_freq='epoch', profile_batch=(1,5))
    model_ckpt_callback = ModelCheckpoint(filepath='models/' + log_dir.split('/')[-1], save_freq='epoch',
                                          monitor='val_accuracy', save_best_only=True),
    clr_callback = CyclicLR(base_lr=hparams[LEARNING_RATE_RANGE], max_lr=hparams[LEARNING_RATE_RANGE] * 5,
                            step_size=100, mode='exp_range', gamma=0.99994)
    lr_log_callback = LRTensorBoard(log_dir=log_dir, update_freq='epoch', profile_batch=0)
    hp_callback = hp.KerasCallback(log_dir, hparams)
    es_callback = EarlyStopping(monitor='val_accuracy', verbose=1, patience=10, mode='max')
    cm_callback = CMTensorboard(log_dir=log_dir, update_freq='epoch', eval_data=eval_data, profile_batch=0)
    custom_model.fit(train_data, validation_data=eval_data, epochs=5,
                     callbacks=[tensorboard_callback, model_ckpt_callback, clr_callback,
                                lr_log_callback, cm_callback, hp_callback, es_callback])
