from tensorflow.python.keras.callbacks import TensorBoard


class LrLog(TensorBoard):
    def __init__(self, log_dir, update_freq):
        super().__init__(log_dir=log_dir, update_freq=update_freq, histogram_freq=0, write_graph=False,
                         write_images=False, profile_batch=0, embeddings_freq=0, embeddings_metadata=None)

    def on_epoch_end(self, epoch, logs=None):
        if 'lr' not in logs.keys():
            logs.update({'lr': self.model.optimizer.lr})
        super(LrLog, self).on_epoch_end(epoch=epoch, logs=logs)
