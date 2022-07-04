from contextlib import redirect_stdout
import tensorflow as tf
import tensorflow_addons as tfa

from data_prep import get_dataset
from features_def import TASK_CLASSES
from custom_metrics import GeometricMean
# from custom_callbacks import EnrTensorboard
from custom_losses import losses


def train_fn(args: dict, dirs: dict, model, strategy):
    """Setup and run training stage"""
    optimizer = {'adam': tf.keras.optimizers.Adam, 'adamax': tf.keras.optimizers.Adamax,
                 'nadam': tf.keras.optimizers.Nadam, 'ftrl': tf.keras.optimizers.Ftrl,
                 'rmsprop': tf.keras.optimizers.RMSprop, 'sgd': tf.keras.optimizers.SGD,
                 'adagrad': tf.keras.optimizers.Adagrad, 'adadelta': tf.keras.optimizers.Adadelta
                 }[args['optimizer']]
    with strategy.scope():
        loss = losses(args)[args['loss_fn']]
        train_data = get_dataset(args=args, dataset='train', dirs=dirs)
        validation_data = get_dataset(args=args, dataset='validation', dirs=dirs)
        model.compile(loss=loss, optimizer=optimizer(learning_rate=args['learning_rate'] * args['gpus']),
                      metrics=[tfa.metrics.F1Score(num_classes=len(TASK_CLASSES[args['task']]), average='macro', name='f1'),
                               GeometricMean()])

    with redirect_stdout(open(dirs['model_summary'], 'w', encoding='utf-8')):
        model.summary()  # show_trainable=True)

    model.fit(x=train_data,
              use_multiprocessing=False, workers=1,
              validation_data=validation_data,
              epochs=args['epochs'], verbose=2,
              callbacks=[tf.keras.callbacks.CSVLogger(filename=dirs['train_logs'], separator=',', append=True),
                         tf.keras.callbacks.EarlyStopping(monitor='val_geometric_mean', mode='max', verbose=1,
                                                          patience=args['early_stop'], restore_best_weights=True),
                         # EnrTensorboard(val_data=val_data, log_dir=dirs['logs'],
                         #                class_names=TASK_CLASSES[args['task']])
                         ]
              )
    model.save(filepath=dirs['save_path'])
