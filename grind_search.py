import os
from datetime import datetime
from hyperparameters import hp_dict
from train_model import training


def grind(args):
    hparam = hp_dict(args=args)
    run_num = 0
    for lr in hparam["learning_rate"].domain.values:
        for batch in hparam["batch_size"].domain.values:
            for img_size in hparam["image_size"].domain.values:
                for do_rate in hparam["dropout_rate"].domain.values:
                    for relu_grad in hparam["relu_grad"].domain.values:
                        for optimizer in hparam["optimizer"].domain.values:
                            for model in hparam["model"].domain.values:
                                hparams = {"lr": {hparam["learning_rate"]: lr},
                                           "batch_size": {hparam["batch_size"]: batch},
                                           "img_size": {hparam["image_size"]: img_size},
                                           "dropout_rate": {hparam["dropout_rate"]: do_rate},
                                           "relu_grad": {hparam["relu_grad"]: relu_grad},
                                           "optimizer": {hparam["optimizer"]: optimizer},
                                           "model": {hparam["model"]: model}}
                                log_dir = f'logs/run-{str(run_num).zfill(4)}-{datetime.now().strftime("%d%m%y%H%M%S")}'
                                if not os.path.exists(log_dir):
                                    os.makedirs(log_dir)
                                with open(log_dir + '/hyperparams.txt', 'a') as f:
                                    hparams_dict = {h : hparams[h].values() for h in hparams}
                                    print(hparams_dict, file=f)
                                # break
                                training(args=args, hparams=hparams, log_dir=log_dir, mode=args.mode)
                                run_num += 1
1