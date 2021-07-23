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
from callbacks import EnrTensorboard, TestCallback, CyclicLR


def training(args, hparams, dir_dict):
    tf.random.set_seed(1312)
    assert args["nodes"] in ("multi", "one")
    if args["nodes"] == 'multi':
        slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
        dir_dict["save_path"] += f"-{slurm_resolver.task_type}-{slurm_resolver.task_type}"
        strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()

    CLASSES = CLASS_NAMES[args["mode"]]
    NUM_CLASSES = len(CLASSES)
    # --------------------------------------------------- Dataset ---------------------------------------------------- #
    hp_key = {key.name: key for key in hparams.keys()}
    global_batch = hparams[hp_key["batch_size"]] * strategy.num_replicas_in_sync
    input_shape = (hparams[hp_key["image_size"]], hparams[hp_key["image_size"]], 3)
    datasets = MelData(dir_dict=dir_dict, args=args, input_shape=input_shape,  batch=global_batch)
    weights = datasets.get_class_weights()
    with open(dir_dict["hparams_logs"], 'a') as f:
        f.write(datasets.info() + f"Epochs: {args['epochs']}\n"
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
                                input_shape=(hparams[hp_key["image_size"]], hparams[hp_key["image_size"]], 3),
                                dropout_rate=hparams[hp_key["dropout"]], alpha=hparams[hp_key["relu_grad"]], classes=NUM_CLASSES)
        custom_model.compile(optimizer=optimizer[hparams[hp_key["optimizer"]]](learning_rate=lr),
                             loss=custom_loss(weights=weights),
                             metrics=metrics(NUM_CLASSES))
    # --------------------------------------------------- Callbacks --------------------------------------------------- #
    steps_per_epoch = np.ceil(datasets.dataset_attributes()["train_len"] / hparams[hp_key["batch_size"]])
    callbacks = [ModelCheckpoint(filepath=dir_dict["save_path"], save_best_only=True),
                 EnrTensorboard(data=datasets.get_dataset(mode='val'), class_names=CLASSES, log_dir=dir_dict["logs"],
                                update_freq='epoch', profile_batch=0, mode=args["mode"]),
                 KerasCallback(writer=dir_dict["logs"], hparams=hparams, trial_id=os.path.basename(dir_dict["trial"])),
                 TestCallback(test_data=datasets.get_dataset(mode="test"), val_data=datasets.get_dataset(mode='val'), mode=args["mode"],
                              class_names=CLASSES, num_classes=NUM_CLASSES, weights=weights, dir_dict=dir_dict),
                 CyclicLR(base_lr=lr, max_lr=lr * 2, step_size=steps_per_epoch * 8, mode='exp_range', gamma=0.99),
                 EarlyStopping(verbose=args["verbose"], patience=args["early_stop"]),
                 tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=dir_dict["backup"])]
    # ------------------------------------------------- Train model -------------------------------------------------- #
    custom_model.fit(x=datasets.get_dataset(mode="train"), epochs=args["epochs"],
                     validation_data=datasets.get_dataset(mode="val"),
                     callbacks=callbacks, verbose=args["verbose"])
    tf.keras.backend.clear_session()
    del datasets
