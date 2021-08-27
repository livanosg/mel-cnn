import numpy as np
import tensorflow as tf
from metrics import calc_metrics, cm_image


class EnrTensorboard(tf.keras.callbacks.TensorBoard):
    def __init__(self, val_data, class_names, **kwargs):
        super().__init__(profile_batch=0, **kwargs)
        self.val_data = val_data
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["learning_rate"] = self.model.optimizer.lr

        # Use the model to predict the values from the validation dataset.
        y_true = []
        test_pred = self.model.predict(self.val_data)
        test_pred = np.argmax(test_pred, axis=-1)
        for data in self.val_data:
            y_true.append(np.argmax(data[1]["class"], axis=-1))
        y_true = np.concatenate(y_true)
        # Calculate the confusion matrix
        confmat_image = cm_image(y_true=y_true, y_pred=test_pred, class_names=self.class_names)
        image = tf.image.decode_png(confmat_image, channels=3)
        image = tf.expand_dims(image, 0)
        with super()._val_writer.as_default():
            tf.summary.image("Confusion Matrix", image, step=epoch)
        super().on_epoch_end(epoch=epoch, logs=logs)


class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_end(self, logs=None):
        model = tf.keras.models.load_model(self.args['dir_dict']['model_path'], compile=False)
        calc_metrics(args=self.args, model=model, dataset=self.args['val_data'], dataset_type='val')
        calc_metrics(args=self.args, model=model, dataset=self.args['test_data'], dataset_type='test')
        if self.args['task'] in ('ben_mal', '5cls'):
            calc_metrics(args=self.args, model=model, dataset=self.args['isic20_test'], dataset_type='isic20_test')


class LaterCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, start_at, **kwargs):
        super().__init__(filepath, **kwargs)
        self.start_at = start_at
        print('Start saving after {} epochs.'.format(self.start_at))

    def on_epoch_end(self, epoch, logs=None):
        if self.start_at > epoch + 1:
            pass
        else:
            if self.start_at == epoch + 1:
                print('Epoch {} reached. Start saving'.format(self.start_at))
            super().on_epoch_end(epoch=epoch, logs=logs)
