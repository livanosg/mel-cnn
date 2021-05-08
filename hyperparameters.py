from tensorboard.plugins.hparams import api as hp


# --------------------------======================= HYPERPARAMETERS ======================-----------------------------#

def hp_dict(args):
    return {"model": hp.HParam('model', hp.Discrete(args["model"])),
            "optimizer": hp.HParam('optimizer', hp.Discrete(args["optimizer"])),
            "img_size": hp.HParam('img_size', hp.Discrete(args["image_size"])),
            "colour": hp.HParam('colour', hp.Discrete(args["colour"])),
            "batch_size": hp.HParam('batch_size', hp.Discrete(args["batch_size"])),
            "lr": hp.HParam('lr', hp.Discrete(args["learning_rate"])),
            "dropout": hp.HParam('dropout', hp.Discrete(args["dropout_rate"])),
            "relu_grad": hp.HParam('relu_grad', hp.Discrete(args["relu_grad"]))}
