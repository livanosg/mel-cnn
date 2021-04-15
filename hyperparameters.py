from tensorboard.plugins.hparams import api as hp


# --------------------------======================= HYPERPARAMETERS ======================-----------------------------#

def hp_dict(args):
    print(args.model)
    return {"learning_rate": hp.HParam('learning_rate', hp.Discrete(args.learning_rate)),  # , 1e-5]))
            "batch_size": hp.HParam('batch', hp.Discrete(args.batch_size)),
            "image_size": hp.HParam('image-size', hp.Discrete(args.image_size)),  # , 300, 512]))
            "dropout_rate": hp.HParam('dropout', hp.Discrete(args.dropout_rate)),
            "relu_grad": hp.HParam('relu_grad', hp.Discrete(args.relu_grad)),  # , 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]))
            "optimizer": hp.HParam('optimizer', hp.Discrete(args.optimizer)),
            "model": hp.HParam('model', hp.Discrete(args.model))}
