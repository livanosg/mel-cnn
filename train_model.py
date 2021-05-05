import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint  # , EarlyStopping
from tensorboard.plugins.hparams.api import KerasCallback
from callbacks import EnrTensorboard  # , CyclicLR
from dataset import MelData
from model import model_fn
from losses import weighted_categorical_crossentropy
from metrics import metrics


def training(args, hparams, hp_keys, log_dir, mode="singledevice"):
    assert mode in ("multinode", "onenode", "onedevice")
    if args.verbose >= 2:
        verbose = 1
    elif args.verbose == 1:
        verbose = 2
    else:
        verbose = 0

    save_path = "models/" + log_dir.split("/")[-1] + "-{epoch:03d}"
    if mode == 'multinode':
        slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
        save_path += f"-{slurm_resolver.task_type}-{slurm_resolver.task_type}"
        strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
    elif mode == 'onedevice':
        strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
    else:
        strategy = tf.distribute.MirroredStrategy()
    os.system(f"echo 'Number of replicas in sync: {strategy.num_replicas_in_sync}'")

    # --------------------------------------------------- Dataset ---------------------------------------------------- #
    global_batch = hparams[hp_keys[4]] * strategy.num_replicas_in_sync
    datasets = MelData(file="all_data_init.csv", frac=args.dataset_frac,
                       batch=global_batch,
                       hw=hparams[hp_keys[2]],
                       colour=hparams[hp_keys[3]])

    # ---------------------------------------------------- Model ----------------------------------------------------- #

    # steps_per_epoch = math.ceil(len(datasets.train_data["class"]) / hparams[hp_keys[4]])
    lr = hparams[hp_keys[5]] * strategy.num_replicas_in_sync
    optimizer = {"adam": tf.keras.optimizers.Adam, "ftrl": tf.keras.optimizers.Ftrl,
                 "sgd": tf.keras.optimizers.SGD, "rmsprop": tf.keras.optimizers.RMSprop,
                 "adadelta": tf.keras.optimizers.Adadelta, "adagrad": tf.keras.optimizers.Adagrad,
                 "adamax": tf.keras.optimizers.Adamax, "nadam": tf.keras.optimizers.Nadam}
    with strategy.scope():
        custom_model = model_fn(model=hparams[hp_keys[0]], input_shape=(hparams[hp_keys[2]], hparams[hp_keys[2]], 3),
                                dropout_rate=hparams[hp_keys[6]], alpha=hparams[hp_keys[7]])
        custom_model.compile(optimizer=optimizer[hparams[hp_keys[1]]](learning_rate=lr),
                             loss=weighted_categorical_crossentropy(weights=datasets.get_class_weights()),
                             metrics=metrics())

    # ---------------------------------------------------- Config ---------------------------------------------------- #
    callbacks = [ModelCheckpoint(filepath=save_path, save_best_only=True),
                 EnrTensorboard(dataclass=datasets, log_dir=log_dir,
                                update_freq='epoch', profile_batch=(5, 15)),
                 KerasCallback(writer=log_dir, hparams=hparams),
                 # CyclicLR(base_lr=lr, max_lr=lr * 5, step_size=steps_per_epoch * 5, mode='exp_range', gamma=0.999),
                 # EarlyStopping(verbose=1, patience=args.early_stop),
                 ]
    if mode != 'singledevice':
        callbacks.append(tf.keras.callbacks.experimental.BackupAndRestore(log_dir + '/tmp'))

    with open(log_dir + '/hyperparams.txt', 'a') as f:
        print(f"{args}", file=f)
        print(f"Train length: {len(datasets.train_data['class'])}\n"
              f"Eval length: {len(datasets.val_data['class'])}", file=f)
        print(f"Weights per class: {datasets.get_class_weights()}\n", file=f)

    # ------------------------------------------------- Train model -------------------------------------------------- #
    custom_model.fit(x=datasets.get_dataset(mode="train", repeat=1), epochs=200, shuffle=False,
                     validation_data=datasets.get_dataset(mode="val", repeat=1),
                     callbacks=callbacks, verbose=verbose)
    tf.keras.backend.clear_session()
