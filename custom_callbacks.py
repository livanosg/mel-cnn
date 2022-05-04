import numpy as np
import tensorflow as tf
from custom_metrics import calc_metrics, cm_image


class EnrTensorboard(tf.keras.callbacks.TensorBoard):
    def __init__(self, val_data, class_names, **kwargs):
        super().__init__(profile_batch=0, write_steps_per_second=True, **kwargs)
        self.val_data = val_data
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # logs["learning_rate"] = self.model.optimizer.lr

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
    def __init__(self, args, validation, test, isic20_test):
        super().__init__()
        self.args = args
        self.validation = validation
        self.test = test
        self.isic20_test = isic20_test

    def on_train_end(self, logs=None):
        model = tf.keras.models.load_model(self.args['dir_dict']['model_path'], compile=False)
        calc_metrics(args=self.args, model=model, dataset=self.validation, dataset_name='validation')
        calc_metrics(args=self.args, model=model, dataset=self.test, dataset_name='test')
        if self.args['task'] in ('ben_mal', '5cls'):
            calc_metrics(args=self.args, model=model, dataset=self.isic20_test, dataset_name='isic20_test')


class LaterCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, start_at, **kwargs):
        super().__init__(**kwargs)
        self.start_at = start_at

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 >= self.start_at:
            if self.start_at == epoch + 1:
                print('Epoch {} reached. Start saving'.format(self.start_at))
            super().on_epoch_end(epoch=epoch, logs=logs)


class LaterReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self, start_at, **kwargs):
        super().__init__(**kwargs)
        self.start_at = start_at

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 >= self.start_at:
            if self.start_at == epoch + 1:
                print('Epoch {} reached. Start checking lr'.format(self.start_at))
            super().on_epoch_end(epoch=epoch, logs=logs)