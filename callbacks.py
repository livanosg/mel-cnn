import itertools
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.callbacks import Callback, TensorBoard, ModelCheckpoint

from metrics import calc_metrics, cm_image


class EnrTensorboard(TensorBoard):
    def __init__(self, data, mode, class_names, **kwargs):
        super().__init__(**kwargs)
        self.eval_data = data
        self.mode = mode
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["learning_rate"] = self.model.optimizer.lr

        # Use the model to predict the values from the validation dataset.
        y_true = []
        test_pred = self.model.predict(self.eval_data)
        test_pred = np.argmax(test_pred, axis=1)
        for data in self.eval_data:
            y_true.append(np.argmax(data[1]["class"], axis=1))
        y_true = np.concatenate(y_true)
        # Calculate the confusion matrix
        confmat_image = cm_image(y_true=y_true, y_pred=test_pred, class_names=self.class_names)
        image = tf.image.decode_png(confmat_image, channels=3)
        image = tf.expand_dims(image, 0)
        with super()._val_writer.as_default():
            tf.summary.image("Confusion Matrix", image, step=epoch)
        super().on_epoch_end(epoch=epoch, logs=logs)


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with some constant frequency,
    as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or per-cycle basis. This class has three
    built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally, it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr and some scaling of the amplitude; therefore max_lr may not
             actually be reached depending on scaling function.
        step_size: number of training iterations per half cycle. Authors suggest setting
        step_size 2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
        Default 'triangular'.
        Values correspond to policies detailed above. If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single argument lambda function,
        where 0 <= scale_fn(x) <= 1 for all x >= 0.
            mode parameter is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on cycle number or cycle iterations (training iterations
            since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs=None):
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        K.set_value(self.model.optimizer.lr, self.clr())


class TestCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_end(self, logs=None):
        model = tf.keras.models.load_model(self.args['dir_dict']['save_path'])
        calc_metrics(args=self.args, model=model, dataset=self.args['test_data'], dataset_type='test')
        calc_metrics(args=self.args, model=model, dataset=self.args['val_data'], dataset_type='val')


class LaterCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, start_at, **kwargs):
        super().__init__(filepath, **kwargs)
        self.start_at = start_at
        print(f'Start saving at {self.start_at}')

    def on_epoch_end(self, epoch, logs=None):
        if self.start_at > epoch:
            pass
        else:
            if self.start_at == epoch:
                print(f'Epoch {epoch} more than {self.start_at}. Start saving')
            super().on_epoch_end(epoch=epoch, logs=logs)
