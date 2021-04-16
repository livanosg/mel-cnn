import os
from datetime import datetime
from hyperparameters import hyperparameters_list
from train_model import training


def grind(args):
    hp_keys = hyperparameters_list(args=args)
    run_num = 0
    for model in hp_keys[0].domain.values:
        for optimizer in hp_keys[1].domain.values:
            for img_size in hp_keys[2].domain.values:
                for batch in hp_keys[3].domain.values:
                    for lr in hp_keys[4].domain.values:
                        for do_rate in hp_keys[5].domain.values:
                            for relu_grad in hp_keys[6].domain.values:
                                hparams = {hp_keys[0]: model,
                                           hp_keys[1]: optimizer,
                                           hp_keys[2]: img_size,
                                           hp_keys[3]: batch,
                                           hp_keys[4]: lr,
                                           hp_keys[5]: do_rate,
                                           hp_keys[6]: relu_grad}
                                log_dir = f'logs/run-{str(run_num).zfill(4)}-{datetime.now().strftime("%d%m%y%H%M%S")}'
                                if not os.path.exists(log_dir):
                                    os.makedirs(log_dir)
                                with open(log_dir + '/hyperparams.txt', 'a') as f:
                                    dicts = {key: hparams[key] for key in hparams}
                                    print(dicts, file=f)
                                training(args=args, hparams=hparams, hp_keys=hp_keys, log_dir=log_dir, mode=args.mode)
                                run_num += 1
