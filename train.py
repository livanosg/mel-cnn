import tensorflow as tf
import tensorflow_addons as tfa
from features_def import TASK_CLASSES
from data_prep import MelData
from custom_metrics import GeometricMean
from custom_callbacks import EnrTensorboard  # , LaterCheckpoint, LaterReduceLROnPlateau
from custom_losses import losses


def train_fn(args, dirs, model, strategy):
    """Setup and run training stage"""
    data = MelData(args, dirs)
    loss = losses(args)[args['loss_fn']]

    with strategy.scope():
        optimizer = {'adam': tf.keras.optimizers.Adam,
                     'adamax': tf.keras.optimizers.Adamax,
                     'nadam': tf.keras.optimizers.Nadam,
                     'ftrl': tf.keras.optimizers.Ftrl,
                     'rmsprop': tf.keras.optimizers.RMSprop,
                     'sgd': tf.keras.optimizers.SGD,
                     'adagrad': tf.keras.optimizers.Adagrad,
                     'adadelta': tf.keras.optimizers.Adadelta}[args['optimizer']]
        model.compile(loss=loss, optimizer=optimizer(learning_rate=args['learning_rate'] * args['gpus']),
                      metrics=[tfa.metrics.F1Score(num_classes=len(TASK_CLASSES[args['task']]),
                                                   average='macro', name='f1'),
                               GeometricMean(num_classes=len(TASK_CLASSES[args['task']]))])

    with open(dirs['model_summary'], 'w', encoding='utf-8') as model_summary:
        model.summary(print_fn=lambda x: model_summary.write(x + '\n'))
    train_data = data.get_dataset(dataset_name='train')
    val_data = data.get_dataset(dataset_name='validation')
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=args['early_stop'], verbose=1, monitor='val_geometric_mean',
                                                  mode='max', restore_best_weights=True),
                 tf.keras.callbacks.CSVLogger(filename=dirs['train_logs'],
                                              separator=',', append=True),
                 EnrTensorboard(val_data=val_data, log_dir=dirs['logs'],
                                class_names=TASK_CLASSES[args['task']])]
    model.fit(x=train_data, validation_data=val_data, callbacks=callbacks,
              epochs=args['epochs'])  # steps_per_epoch=np.floor(data.train_len / batch),
    model.save(filepath=dirs['save_path'])
    return model
