import math
import os
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorboard.plugins.hparams.api import KerasCallback
from dataset import MelData
from model import model_fn
from metrics import metrics
from callbacks import EnrTensorboard, CyclicLR


def training(args, hparams, hp_keys, log_dir, mode="singledevice"):

    assert mode in ("multinode", "singlenode", "singledevice")
    os.system(f"echo 'Running mode: {mode}.'")
    save_path = "models/" + log_dir.split("/")[-1] + "-{epoch:03d}"
    if mode == 'multinode':
        slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
        save_path += f"-{slurm_resolver.task_type}-{slurm_resolver.task_type}"
        strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
    elif mode == 'singledevice':
        strategy = tf.distribute.OneDeviceStrategy(device=':/gpu')
    else:
        strategy = tf.distribute.MirroredStrategy()
    os.system(f"echo 'Number of replicas in sync: {strategy.num_replicas_in_sync}'")

    models = {'xept': (applications.xception.Xception, applications.xception.preprocess_input),
              'incept': (applications.inception_v3.InceptionV3, applications.inception_v3.preprocess_input),
              'effnet0': (applications.efficientnet.EfficientNetB0, applications.efficientnet.preprocess_input),
              'effnet1': (applications.efficientnet.EfficientNetB1, applications.efficientnet.preprocess_input)}

    # DATA
    datasets = MelData(size=args.dataset_size,
                       batch_size=hparams[hp_keys[3]] * strategy.num_replicas_in_sync,
                       hwc=hparams[hp_keys[2]])
    with strategy.scope():
        custom_model = model_fn(model_list=models[hparams[hp_keys[0]]],
                                input_shape=(hparams[hp_keys[2]], hparams[hp_keys[2]], 3),
                                dropout_rate=hparams[hp_keys[5]],
                                alpha=hparams[hp_keys[6]])
        custom_model.compile(optimizer=hparams[hp_keys[1]],
                             loss='categorical_crossentropy',
                             loss_weights={'classes': datasets.get_class_weights()},
                             metrics=metrics())
    steps_per_epoch = math.ceil(datasets.train_len / hparams[hp_keys[3]])
    lr = hparams[hp_keys[4]] * strategy.num_replicas_in_sync
    # validation_steps = math.ceil(datasets.eval_len / hparams[BATCH_SIZE_RANGE])
    callbacks = [ModelCheckpoint(filepath=save_path, save_best_only=True),
                 EnrTensorboard(eval_data=datasets.get_dataset('eval', 1), log_dir=log_dir, update_freq='epoch', profile_batch=0),
                 KerasCallback(writer=log_dir, hparams=hparams),
                 CyclicLR(base_lr=lr, max_lr=lr * 5, step_size=steps_per_epoch * 5, mode='exp_range', gamma=0.999),
                 EarlyStopping(verbose=1, patience=args.early_stop)]
    if mode != 'singledevice':
        callbacks.append(tf.keras.callbacks.experimental.BackupAndRestore(log_dir + '/tmp'))

    with open(log_dir + '/hyperparams.txt', 'a') as f:
        print(datasets.get_class_weights(), file=f)

    os.system(f"echo 'Train length: {datasets.train_len} | Eval length: {datasets.eval_len}'\n"
              f"echo 'Weights per class: {datasets.get_class_weights()}'")
    if args.verbose >= 2:
        verbose = 1
    elif args.verbose == 1:
        verbose = 2
    else:
        verbose = 0
    custom_model.fit(x=datasets.get_dataset('train', repeat=1), epochs=500, shuffle=False,
                     validation_data=datasets.get_dataset('eval', repeat=1),
                     callbacks=callbacks, verbose=verbose)
    tf.keras.backend.clear_session()
