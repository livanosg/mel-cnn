import math
import os.path

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorboard.plugins.hparams.api import KerasCallback
from dataset import MelData
from model import model_fn
from losses import weighted_categorical_crossentropy
from metrics import metrics
from callbacks import EnrTensorboard, CyclicLR


def training(args, hparams, dir_dict):
    tf.random.set_seed(1312)
    if args["binary"] == "binary":
        binary = True
    else:
        binary = False

    assert args["nodes"] in ("multi", "one")
    save_path = dir_dict["save_path"]
    if args["nodes"] == 'multi':
        slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
        save_path += f"-{slurm_resolver.task_type}-{slurm_resolver.task_type}"
        strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()
    # --------------------------------------------------- Dataset ---------------------------------------------------- #
    hp_key = {key.name: key for key in hparams.keys()}
    global_batch = hparams[hp_key["batch_size"]] * strategy.num_replicas_in_sync
    datasets = MelData(file="all_data_init.csv", frac=args["dataset_frac"],
                       batch=global_batch, img_folder=dir_dict["image_folder"], binary=binary)
    data_info = datasets.data_info()
    with open(dir_dict["trial_config"], 'a') as f:
        print(f"Epochs: {args['epochs']}\n"
              f"Early-stop: {args['early_stop']}\n"
              f"Nodes: {args['nodes']}\n"
              f"Train length: {data_info['train_len']}\n"
              f"Eval length: {data_info['val_len']}\n"
              f"Weights per class: {datasets.get_class_weights()}\n"
              f"Number of replicas in sync: {strategy.num_replicas_in_sync}", file=f)
    # ---------------------------------------------------- Model ----------------------------------------------------- #
    steps_per_epoch = math.ceil(data_info["train_len"] / hparams[hp_key["batch_size"]])
    lr = hparams[hp_key["lr"]] * strategy.num_replicas_in_sync
    optimizer = {"adam": tf.keras.optimizers.Adam, "ftrl": tf.keras.optimizers.Ftrl,
                 "sgd": tf.keras.optimizers.SGD, "rmsprop": tf.keras.optimizers.RMSprop,
                 "adadelta": tf.keras.optimizers.Adadelta, "adagrad": tf.keras.optimizers.Adagrad,
                 "adamax": tf.keras.optimizers.Adamax, "nadam": tf.keras.optimizers.Nadam}
    with strategy.scope():
        custom_model = model_fn(model=hparams[hp_key["model"]], input_shape=(hparams[hp_key["img_size"]], hparams[hp_key["img_size"]], 3),
                                dropout_rate=hparams[hp_key["dropout"]], alpha=hparams[hp_key["relu_grad"]], binary=binary)
        custom_model.compile(optimizer=optimizer[hparams[hp_key["optimizer"]]](learning_rate=lr),
                             loss=weighted_categorical_crossentropy(weights=datasets.get_class_weights()),
                             metrics=metrics())
    # ---------------------------------------------------- Config ---------------------------------------------------- #
    callbacks = [ModelCheckpoint(filepath=save_path, save_best_only=True),
                 EnrTensorboard(dataclass=datasets, log_dir=dir_dict["logs"], update_freq='epoch', profile_batch=0),
                 KerasCallback(writer=dir_dict["logs"], hparams=hparams, trial_id=os.path.basename(dir_dict["trial"])),
                 CyclicLR(base_lr=lr, max_lr=lr * 5, step_size=steps_per_epoch * 8, mode='exp_range', gamma=0.999),
                 EarlyStopping(verbose=1, patience=args["early_stop"]),
                 tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=dir_dict["backup"])]
    # ------------------------------------------------- Train model -------------------------------------------------- #
    if args["verbose"] == 1:
        verbose = 2
    elif args["verbose"] >= 2:
        verbose = 1
    else:
        verbose = 0
    custom_model.fit(x=datasets.get_dataset(mode="train", repeat=1), epochs=200, shuffle=False,
                     validation_data=datasets.get_dataset(mode="val", repeat=1),
                     callbacks=callbacks, verbose=verbose)
    tf.keras.backend.clear_session()
