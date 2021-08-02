import tensorflow as tf
from tensorflow.keras.applications import xception, inception_v3, efficientnet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks.experimental import BackupAndRestore
from config import CLASS_NAMES
from dataset import MelData
from model import model_fn
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from metrics import metrics
from callbacks import EnrTensorboard, TestCallback, LaterCheckpoint


def training(args):
    tf.random.set_seed(5)
    assert args["nodes"] in ("multi", "one")
    if args["nodes"] == 'multi':
        slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
        args["dir_dict"]["save_path"] += f"-{slurm_resolver.task_type}-{slurm_resolver.task_type}"
        strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()

    args['class_names'] = CLASS_NAMES[args["mode"]]
    args['num_classes'] = len(args['class_names'])
    models = {'xept': xception.Xception, 'incept': inception_v3.InceptionV3,
              'effnet0': efficientnet.EfficientNetB0, 'effnet1': efficientnet.EfficientNetB1}

    preproc_input_fn = {'xept':  xception.preprocess_input, 'incept': inception_v3.preprocess_input,
                        'effnet0': efficientnet.preprocess_input, 'effnet1':  efficientnet.preprocess_input}
    args['preprocess_fn'] = preproc_input_fn[args['model']]
    args['model'] = models[args['model']]
    # --------------------------------------------------- Dataset ---------------------------------------------------- #
    global_batch = args['batch_size'] * strategy.num_replicas_in_sync
    args['input_shape'] = (args['image_size'], args['image_size'], 3)
    # ---------------------------------------------------- Model ----------------------------------------------------- #
    args["learning_rate"] = args["learning_rate"] * strategy.num_replicas_in_sync
    optimizer = {"adam": tf.keras.optimizers.Adam, "ftrl": tf.keras.optimizers.Ftrl,
                 "sgd": tf.keras.optimizers.SGD, "rmsprop": tf.keras.optimizers.RMSprop,
                 "adadelta": tf.keras.optimizers.Adadelta, "adagrad": tf.keras.optimizers.Adagrad,
                 "adamax": tf.keras.optimizers.Adamax, "nadam": tf.keras.optimizers.Nadam}
    with strategy.scope():
        datasets = MelData(dir_dict=args['dir_dict'], args=args, batch=global_batch)
        train_data = datasets.get_dataset(mode='train')
        val_data = datasets.get_dataset(mode='val')
        test_data = datasets.get_dataset(mode='test')
        args['weights'] = datasets.weights_per_class
        with open(args['dir_dict']['hparams_logs'], 'a') as f:
            f.write(datasets.info() + f"Number of replicas in sync: {strategy.num_replicas_in_sync}\n")

        custom_model = model_fn(args=args)
        custom_model.compile(optimizer=optimizer[args["optimizer"]](learning_rate=args["learning_rate"]),
                             loss=SigmoidFocalCrossEntropy(gamma=2.5, alpha=0.2, reduction=tf.keras.losses.Reduction.AUTO),  #'categorical_crossentropy',  # custom_loss(datasets.weights_per_class)
                             metrics=metrics(args['num_classes']))
    # --------------------------------------------------- Callbacks --------------------------------------------------- #
    callbacks = [LaterCheckpoint(filepath=args["dir_dict"]["save_path"], save_best_only=True, start_at=20),
                 EnrTensorboard(data=val_data, class_names=args['class_names'], log_dir=args["dir_dict"]["logs"],
                                update_freq='epoch', profile_batch=0, mode=args["mode"]),
                 TestCallback(test_data=test_data, val_data=val_data, args=args),
                 ReduceLROnPlateau(factor=0.75, patience=10),
                 EarlyStopping(verbose=args["verbose"], patience=args["early_stop"]),
                 BackupAndRestore(backup_dir=args["dir_dict"]["backup"])]
    # ------------------------------------------------- Train model -------------------------------------------------- #
    custom_model.fit(x=train_data, epochs=args["epochs"],
                     validation_data=val_data,
                     callbacks=callbacks, verbose=args["verbose"])
    tf.keras.backend.clear_session()
