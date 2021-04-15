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


def training(args, hparams, log_dir, mode="singledevice"):
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
    datasets = MelData(size=args.dataset_size, batch_size=list(hparams["batch_size"].values())[0] * strategy.num_replicas_in_sync,
                       hwc=list(hparams["img_size"].values())[0])
    with strategy.scope():
        # MODEL
        # -----------------------------================ Image part =================---------------------------------- #
        custom_model = model_fn(model_list=models[list(hparams["model"].values())[0]],
                                input_shape=(list(hparams["img_size"].values())[0], list(hparams["img_size"].values())[0], 3),
                                dropout_rate=list(hparams["dropout_rate"].values())[0],
                                alpha=list(hparams["relu_grad"].values())[0])
        custom_model.compile(optimizer=list(hparams["optimizer"].values())[0],
                             loss='categorical_crossentropy',
                             loss_weights={'classes': datasets.get_class_weights()},
                             metrics=metrics())
    # TRAIN
    merge = {}
    [merge.update(pair) for pair in hparams.values()]
    steps_per_epoch = math.ceil(datasets.train_len / list(hparams["batch_size"].values())[0])
    # validation_steps = math.ceil(datasets.eval_len / hparams[BATCH_SIZE_RANGE])
    callbacks = [ModelCheckpoint(filepath=save_path, save_best_only=True),
                 EnrTensorboard(eval_data=datasets.get_dataset('eval', 1), log_dir=log_dir, update_freq='epoch', profile_batch=(2, 4)),
                 KerasCallback(writer=log_dir + "/hparams", hparams=merge),
                 CyclicLR(base_lr=list(hparams["lr"].values())[0], max_lr=list(hparams["lr"].values())[0] * 5, step_size=steps_per_epoch * 2, mode='exp_range', gamma=0.999),
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
