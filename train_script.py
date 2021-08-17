import os.path
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from model import model_fn
from callbacks import LaterCheckpoint, EnrTensorboard


def training(args, strategy):
    # --------------------------------------------------- Dataset ---------------------------------------------------- #
    optimizer = {"adam": tf.keras.optimizers.Adam, "ftrl": tf.keras.optimizers.Ftrl,
                 "sgd": tf.keras.optimizers.SGD, "rmsprop": tf.keras.optimizers.RMSprop,
                 "adadelta": tf.keras.optimizers.Adadelta, "adagrad": tf.keras.optimizers.Adagrad,
                 "adamax": tf.keras.optimizers.Adamax, "nadam": tf.keras.optimizers.Nadam}[args["optimizer"]](learning_rate=args["learning_rate"])

    with strategy.scope():
        custom_model = model_fn(args=args)
        with open(os.path.join(args['dir_dict']['trial'], 'model_summary.txt'), 'w') as f:
            custom_model.summary(print_fn=lambda x: f.write(x + '\n'))

        custom_model.compile(optimizer=optimizer,
                             loss='categorical_crossentropy',  # SigmoidFocalCrossEntropy(gamma=2.5, alpha=0.2, reduction=tf.keras.losses.Reduction.AUTO), # custom_loss(datasets.weights_per_class)
                             metrics=[AUC(multi_label=True)])
        # --------------------------------------------------- Callbacks --------------------------------------------------- #
        callbacks = [LaterCheckpoint(filepath=args["dir_dict"]["save_path"], save_best_only=True, start_at=0),
                     EnrTensorboard(log_dir=args["dir_dict"]["logs"], val_data=args['val_data'], class_names=args['class_names'], profile_batch=0),
                     ReduceLROnPlateau(factor=0.75, patience=10),
                     EarlyStopping(verbose=args["verbose"], patience=args["early_stop"])]
        # ------------------------------------------------- Train model -------------------------------------------------- #
        custom_model.fit(x=args['train_data'], epochs=args["epochs"],
                         validation_data=args['val_data'],
                         callbacks=callbacks, verbose=args["verbose"])

