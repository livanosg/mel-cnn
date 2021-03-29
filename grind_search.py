import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.python.data.experimental import AutoShardPolicy
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.xception import Xception

from dataset import MelData
from hyperparam import HWC_RANGE, BATCH_SIZE_RANGE, DROPOUT_RANGE, ACTIVATION_OPTIONS, HP_OPTIMIZER, \
    LEARNING_RATE_RANGE, HP_MODELS, METRIC_ACCURACY


def training(hparams, log_dir):
    distr = tf.distribute.OneDeviceStrategy(device='/cpu')  # Testing distribution operations
    if model == 'xception':
        # tf.keras.applications.xception.preprocess_input()
       pass
    elif model == 'inception':
        # tf.keras.applications.inception_v3.preprocess_input()
        pass
    else:
        # tf.keras.applications.efficientnet.preprocess_input()
        pass
    with distr.scope():
        image_input = keras.Input(shape=(hparams[HWC_RANGE], hparams[HWC_RANGE], 3), batch_size=hparams[BATCH_SIZE_RANGE], name='image')
        images_base_model = models[hparams[HP_MODELS]](include_top=False, input_tensor=image_input)
        images_base_model.trainable = False
        reduce_base_model = keras.layers.Conv2D(100, kernel_size=1, padding='same')(images_base_model.output)
        flattened_basemodel_output = keras.layers.Flatten()(reduce_base_model)
        image_model_fcl_1 = keras.layers.Dense(int(flattened_basemodel_output.shape[-1] / 2), tf.keras.layers.LeakyReLU(alpha=hparams[ACTIVATION_OPTIONS]))(flattened_basemodel_output)
        image_model_fcl_2 = keras.layers.Dense(int(image_model_fcl_1.shape[-1] / 2), tf.keras.layers.LeakyReLU(alpha=hparams[ACTIVATION_OPTIONS]))(image_model_fcl_1)
        output_layer = keras.layers.Dense(5, activation='softmax', name='classes')(image_model_fcl_2)
        custom_model = tf.keras.Model(image_input, output_layer)
        custom_model.compile(hparams[HP_OPTIMIZER], loss='categorical_crossentropy', metrics=['accuracy'])

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    options.experimental_threading.max_intra_op_parallelism = 1

    train_data, eval_data = MelData(hparams=hparams)
    train_data = train_data.with_options(options)
    eval_data = eval_data.with_options(options)

    custom_model.fit(train_data, validation_data=eval_data, epochs=5,
                     callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=10, profile_batch=(2, 4)),
                                hp.KerasCallback(log_dir, hparams)])



models = {'xception': Xception, 'inception': InceptionV3, 'efficientnet0': EfficientNetB0,
          'efficientnet1': EfficientNetB1}

run_num = 0
for LR in LEARNING_RATE_RANGE.domain.values:
    for batch_size in BATCH_SIZE_RANGE.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            for HWC in HWC_RANGE.domain.values:
                for activation_a in ACTIVATION_OPTIONS.domain.values:
                    for model in HP_MODELS.domain.values:
                        for droput_rate in (DROPOUT_RANGE.domain.min_value, DROPOUT_RANGE.domain.max_value):
                            hparams = {LEARNING_RATE_RANGE: LR,
                                       BATCH_SIZE_RANGE: batch_size,
                                       HWC_RANGE: HWC,
                                       ACTIVATION_OPTIONS: activation_a,
                                       DROPOUT_RANGE: droput_rate,
                                       HP_OPTIMIZER: optimizer,
                                       HP_MODELS: model}
                            print({h.name: hparams[h] for h in hparams})
                            training(hparams, log_dir=f'run-{run_num}')
                            run_num += 1
