import math
import os
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorboard.plugins.hparams.api import KerasCallback
from hyperparameters import RELU_A, HWC_DOM, MODEL_LST, OPTIMIZER_LST, LR_LST, DROPOUT_LST, BATCH_SIZE_RANGE, metrics
from model import model_fn
from dataset import MelData
from callbacks import EnrTensorboard, CyclicLR

tf.random.set_seed(0)


def training(hparams, log_dir, nodes='local'):
    assert nodes in ('multi', 'one', 'local')
    os.system(f"echo 'Running {nodes}-node.'")
    save_path = "models/" + log_dir.split("/")[-1] + "-{epoch:03d}"
    if nodes == 'multi':
        slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
        save_path += f"-{slurm_resolver.task_type}-{slurm_resolver.task_type}"
        strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
    elif nodes == 'local':
        strategy = tf.distribute.OneDeviceStrategy(device='cpu')
    else:
        strategy = tf.distribute.MirroredStrategy()
    os.system(f"echo 'Number of replicas in sync: {strategy.num_replicas_in_sync}'")

    models = {'xception': (applications.xception.Xception, applications.xception.preprocess_input),
              'inception': (applications.inception_v3.InceptionV3, applications.inception_v3.preprocess_input),
              'efficientnet0': (applications.efficientnet.EfficientNetB0, applications.efficientnet.preprocess_input),
              'efficientnet1': (applications.efficientnet.EfficientNetB1, applications.efficientnet.preprocess_input)}

    # DATA
    datasets = MelData(size=-1, batch_size=hparams[BATCH_SIZE_RANGE] * strategy.num_replicas_in_sync,
                       hwc=hparams[HWC_DOM])
    with strategy.scope():
        # MODEL
        # -----------------------------================ Image part =================---------------------------------- #
        custom_model = model_fn(model_list=models[hparams[MODEL_LST]],
                                input_shape=(hparams[HWC_DOM], hparams[HWC_DOM], 3),
                                dropout_rate=hparams[DROPOUT_LST],
                                alpha=hparams[RELU_A])
        custom_model.compile(optimizer=hparams[OPTIMIZER_LST],
                             loss='categorical_crossentropy',
                             loss_weights={'classes': datasets.get_class_weights()},
                             metrics=metrics())
    # TRAIN
    steps_per_epoch = math.ceil(datasets.train_len / hparams[BATCH_SIZE_RANGE])
    validation_steps = math.ceil(datasets.eval_len / hparams[BATCH_SIZE_RANGE])
    callbacks = [ModelCheckpoint(filepath=save_path, save_best_only=True),
                 EnrTensorboard(eval_data=datasets.get_dataset('eval', 1), log_dir=log_dir, update_freq='epoch', profile_batch=(2, 4)),
                 KerasCallback(writer=log_dir, hparams=hparams),
                 CyclicLR(base_lr=hparams[LR_LST], max_lr=hparams[LR_LST] * 5, step_size=steps_per_epoch * 2, mode='exp_range', gamma=0.999),
                 EarlyStopping(verbose=1, patience=20)]
    if nodes != 'local':
        callbacks.append(tf.keras.callbacks.experimental.BackupAndRestore(log_dir + '/tmp'))

    with open(log_dir + '/hyperparams.txt', 'a') as f:
        print(datasets.get_class_weights(), file=f)

    os.system(f"echo 'Train length: {datasets.train_len} | Eval length: {datasets.eval_len}'\n"
              f"echo 'Weights per class: {datasets.get_class_weights()}'")
    custom_model.fit(x=datasets.get_dataset('train', repeat=1), epochs=500, steps_per_epoch=steps_per_epoch, shuffle=False,
                     validation_data=datasets.get_dataset('eval', repeat=1), validation_steps=validation_steps,
                     callbacks=callbacks, verbose=2)
    tf.keras.backend.clear_session()
