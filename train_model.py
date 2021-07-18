import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorboard.plugins.hparams.api import KerasCallback

from config import CLASS_NAMES
from dataset import MelData
from model import model_fn
from losses import custom_loss
from metrics import metrics
from callbacks import EnrTensorboard, TestCallback, CyclicLR  # , CyclicLR


def training(args, hparams, dir_dict):
    tf.random.set_seed(1312)
    if args["verbose"] == 1:
        verbose = 2
    elif args["verbose"] >= 2:
        verbose = 1
    else:
        verbose = 0

    assert args["nodes"] in ("multi", "one")
    save_path = dir_dict["save_path"]
    if args["nodes"] == 'multi':
        slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
        save_path += f"-{slurm_resolver.task_type}-{slurm_resolver.task_type}"
        strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()

    CLASSES = CLASS_NAMES[args["mode"]]
    NUM_CLASSES = len(CLASSES)
    # --------------------------------------------------- Dataset ---------------------------------------------------- #
    hp_key = {key.name: key for key in hparams.keys()}
    global_batch = hparams[hp_key["batch_size"]] * strategy.num_replicas_in_sync
    datasets = MelData(file="all_data.csv", trial=dir_dict["trial"], frac=args["dataset_frac"],
                       batch=global_batch, img_folder=dir_dict["image_folder"], mode=args["mode"])
    weights = datasets.get_class_weights()
    with open(dir_dict["trial_config"], 'a') as f:
        f.write(datasets.logs() + f"Epochs: {args['epochs']}\n"
                                  f"Early-stop: {args['early_stop']}\n"
                                  f"Nodes: {args['nodes']}\n"
                                  f"Number of replicas in sync: {strategy.num_replicas_in_sync}\n")
    # ---------------------------------------------------- Model ----------------------------------------------------- #
    lr = hparams[hp_key["lr"]] * strategy.num_replicas_in_sync
    optimizer = {"adam": tf.keras.optimizers.Adam, "ftrl": tf.keras.optimizers.Ftrl,
                 "sgd": tf.keras.optimizers.SGD, "rmsprop": tf.keras.optimizers.RMSprop,
                 "adadelta": tf.keras.optimizers.Adadelta, "adagrad": tf.keras.optimizers.Adagrad,
                 "adamax": tf.keras.optimizers.Adamax, "nadam": tf.keras.optimizers.Nadam}
    with strategy.scope():
        custom_model = model_fn(model=hparams[hp_key["model"]],
                                input_shape=(hparams[hp_key["img_size"]], hparams[hp_key["img_size"]], 3),
                                dropout_rate=hparams[hp_key["dropout"]], alpha=hparams[hp_key["relu_grad"]], classes=NUM_CLASSES)
        custom_model.compile(optimizer=optimizer[hparams[hp_key["optimizer"]]](learning_rate=lr),
                             loss=custom_loss(weights=weights),
                             metrics=metrics(NUM_CLASSES))
    # --------------------------------------------------- Callbacks --------------------------------------------------- #
    steps_per_epoch = np.ceil(datasets.dataset_attributes()["train_len"] / hparams[hp_key["batch_size"]])
    callbacks = [ModelCheckpoint(filepath=save_path, save_best_only=True),
                 EnrTensorboard(data=datasets.get_dataset(mode='val'), class_names=CLASSES, log_dir=dir_dict["logs"],
                                update_freq='epoch', profile_batch=0, mode=args["mode"]),
                 KerasCallback(writer=dir_dict["logs"], hparams=hparams, trial_id=os.path.basename(dir_dict["trial"])),
                 TestCallback(test_data=datasets.get_dataset(mode="test"), mode=args["mode"],
                              trial_path=dir_dict["trial"], class_names=CLASSES, classes=NUM_CLASSES, weights=weights),
                 CyclicLR(base_lr=lr, max_lr=lr * 5, step_size=steps_per_epoch * 8, mode='exp_range', gamma=0.999),
                 EarlyStopping(verbose=verbose, patience=args["early_stop"]),
                 tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=dir_dict["backup"])]
    # ------------------------------------------------- Train model -------------------------------------------------- #
    custom_model.fit(x=datasets.get_dataset(mode="train"), epochs=args["epochs"], shuffle=False,
                     validation_data=datasets.get_dataset(mode="val"),
                     callbacks=callbacks, verbose=verbose)
    tf.keras.backend.clear_session()
