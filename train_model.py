import os
# import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import xception, inception_v3, efficientnet
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorboard.plugins.hparams.api import KerasCallback
from config import CLASS_NAMES
from dataset import MelData
from model import model_fn
from losses import custom_loss
from metrics import metrics
from callbacks import EnrTensorboard, TestCallback  # , CyclicLR


def training(args):
    tf.random.set_seed(1312)
    assert args["nodes"] in ("multi", "one")
    if args["nodes"] == 'multi':
        slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
        args["dir_dict"]["save_path"] += f"-{slurm_resolver.task_type}-{slurm_resolver.task_type}"
        strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()

    args['class_names'] = CLASS_NAMES[args["mode"]]
    args['num_classes'] = len(args['class_names'])
    models = {'xept': (xception.Xception, xception.preprocess_input),
              'incept': (inception_v3.InceptionV3, inception_v3.preprocess_input),
              'effnet0': (efficientnet.EfficientNetB0, efficientnet.preprocess_input),
              'effnet1': (efficientnet.EfficientNetB1, efficientnet.preprocess_input)}

    args["preprocess_fn"] = models[args["model"]][1]
    args['model'] = models[args["model"]][0]
    # --------------------------------------------------- Dataset ---------------------------------------------------- #
    global_batch = args["batch_size"] * strategy.num_replicas_in_sync
    args['input_shape'] = (args["image_size"], args["image_size"], 3)
    datasets = MelData(dir_dict=args["dir_dict"], args=args, batch=global_batch)
    args['weights'] = datasets.weights_per_class()
    with open(args["dir_dict"]["hparams_logs"], 'a') as f:
        f.write(datasets.info() + f"Number of replicas in sync: {strategy.num_replicas_in_sync}\n")
    # ---------------------------------------------------- Model ----------------------------------------------------- #
    args["lr"] = args["lr"] * strategy.num_replicas_in_sync
    optimizer = {"adam": tf.keras.optimizers.Adam, "ftrl": tf.keras.optimizers.Ftrl,
                 "sgd": tf.keras.optimizers.SGD, "rmsprop": tf.keras.optimizers.RMSprop,
                 "adadelta": tf.keras.optimizers.Adadelta, "adagrad": tf.keras.optimizers.Adagrad,
                 "adamax": tf.keras.optimizers.Adamax, "nadam": tf.keras.optimizers.Nadam}
    with strategy.scope():
        custom_model = model_fn(args=args)
        custom_model.compile(optimizer=optimizer[args["optimizer"]](learning_rate=args["lr"]),
                             loss=custom_loss(weights=args['weights']),
                             metrics=metrics(args['num_classes']))
    # --------------------------------------------------- Callbacks --------------------------------------------------- #
    # steps_per_epoch = np.ceil(datasets.dataset_attributes()["train_len"] / args["batch_size"])
    callbacks = [ModelCheckpoint(filepath=args["dir_dict"]["save_path"], save_best_only=True),
                 EnrTensorboard(data=datasets.get_dataset(mode='val'), class_names=args['class_names'], log_dir=args["dir_dict"]["logs"],
                                update_freq='epoch', profile_batch=0, mode=args["mode"]),
                 KerasCallback(writer=args["dir_dict"]["logs"], hparams=args["hparams"], trial_id=os.path.basename(args["dir_dict"]["trial"])),
                 TestCallback(test_data=datasets.get_dataset(mode="test"), val_data=datasets.get_dataset(mode='val'), mode=args["mode"],
                              class_names=args['class_names'], num_classes=args['num_classes'], weights=args['weights'], dir_dict=args["dir_dict"]),
                 # CyclicLR(base_lr=lr, max_lr=lr * 2, step_size=steps_per_epoch * 8, mode='exp_range', gamma=0.99),
                 EarlyStopping(verbose=args["verbose"], patience=args["early_stop"]),
                 tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=args["dir_dict"]["backup"])]
    # ------------------------------------------------- Train model -------------------------------------------------- #
    custom_model.fit(x=datasets.get_dataset(mode="train"), epochs=args["epochs"],
                     validation_data=datasets.get_dataset(mode="val"),
                     callbacks=callbacks, verbose=args["verbose"])
    tf.keras.backend.clear_session()
