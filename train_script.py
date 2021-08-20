import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from model import model_fn
from custom_losses import PerClassWeightedCategoricalCrossentropy
from callbacks import LaterCheckpoint, EnrTensorboard, TestCallback


def training(args, strategy):
    # --------------------------------------------------- Dataset ---------------------------------------------------- #
    optimizer = {"adam": tf.keras.optimizers.Adam, "ftrl": tf.keras.optimizers.Ftrl,
                 "sgd": tf.keras.optimizers.SGD, "rmsprop": tf.keras.optimizers.RMSprop,
                 "adadelta": tf.keras.optimizers.Adadelta, "adagrad": tf.keras.optimizers.Adagrad,
                 "adamax": tf.keras.optimizers.Adamax, "nadam": tf.keras.optimizers.Nadam}[args["optimizer"]](learning_rate=args["learning_rate"] * args['replicas'])

    with strategy.scope():
        custom_model = model_fn(args=args)
        with open(args['dir_dict']['trial'] + '/model_summary.txt', 'w') as f:
            custom_model.summary(print_fn=lambda x: f.write(x + '\n'))

        custom_model.compile(optimizer=optimizer,
                             loss=PerClassWeightedCategoricalCrossentropy(args=args),  # WeightedCategoricalCrossentropy
                             metrics=[AUC(multi_label=True)])
        # --------------------------------------------------- Callbacks --------------------------------------------------- #
        callbacks = [LaterCheckpoint(filepath=args["dir_dict"]["model_path"], save_best_only=True, start_at=25),
                     EnrTensorboard(log_dir=args["dir_dict"]["logs"], val_data=args['val_data'], class_names=args['class_names']),
                     ReduceLROnPlateau(factor=0.75, patience=10),
                     EarlyStopping(verbose=args["verbose"], patience=20),
                     TestCallback(args=args)]
        # ------------------------------------------------- Train model -------------------------------------------------- #
        custom_model.fit(x=args['train_data'], epochs=args["epochs"],
                         validation_data=args['val_data'],
                         callbacks=callbacks, verbose=args["verbose"])
