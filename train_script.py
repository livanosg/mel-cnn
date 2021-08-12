import tensorflow as tf
from tensorflow.keras.applications import xception, inception_v3, efficientnet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_pipe import MelData
from model import model_fn
from metrics import metrics
from callbacks import EnrTensorboard, TestCallback, LaterCheckpoint


def training(args):
    # --------------------------------------------------- Dataset ---------------------------------------------------- #
    optimizer = {"adam": tf.keras.optimizers.Adam, "ftrl": tf.keras.optimizers.Ftrl,
                 "sgd": tf.keras.optimizers.SGD, "rmsprop": tf.keras.optimizers.RMSprop,
                 "adadelta": tf.keras.optimizers.Adadelta, "adagrad": tf.keras.optimizers.Adagrad,
                 "adamax": tf.keras.optimizers.Adamax, "nadam": tf.keras.optimizers.Nadam}

    with args['strategy'].scope():
        datasets = MelData(args=args)
        train_data = datasets.get_dataset(mode='train')
        val_data = datasets.get_dataset(mode='val')
        test_data = datasets.get_dataset(mode='test')
        with open(args['dir_dict']['hparams_logs'], 'a') as f:
            f.write(datasets.info() + f"Number of replicas in sync: {args['strategy'].num_replicas_in_sync}\n")

        custom_model = model_fn(args=args)
        custom_model.compile(optimizer=optimizer[args["optimizer"]](learning_rate=args["learning_rate"]),
                             loss='categorical_crossentropy',  # SigmoidFocalCrossEntropy(gamma=2.5, alpha=0.2, reduction=tf.keras.losses.Reduction.AUTO), # custom_loss(datasets.weights_per_class)
                             metrics=metrics())
    # --------------------------------------------------- Callbacks --------------------------------------------------- #
    callbacks = [LaterCheckpoint(filepath=args["dir_dict"]["save_path"], save_best_only=True, start_at=25),
                 EnrTensorboard(data=val_data, class_names=args['class_names'], log_dir=args["dir_dict"]["logs"],
                                profile_batch=0, mode=args["mode"]),
                 TestCallback(args=args, val_data=val_data, test_data=test_data),
                 ReduceLROnPlateau(factor=0.75, patience=10),
                 EarlyStopping(verbose=args["verbose"], patience=args["early_stop"])]
    # ------------------------------------------------- Train model -------------------------------------------------- #
    custom_model.fit(x=train_data, epochs=args["epochs"],
                     validation_data=val_data,
                     callbacks=callbacks, verbose=args["verbose"])
    tf.keras.backend.clear_session()
