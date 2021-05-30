from datetime import datetime
from config import directories
from hyperparameters import hp_dict
from prep_dataset import check_create_dataset
from train_model import training


def grind(args):
    hp = hp_dict(args=args)
    run_num = 0
    trial_id = datetime.now().strftime('%d%m%y%H%M%S')
    for model in hp["model"].domain.values:
        for optimizer in hp["optimizer"].domain.values:
            for img_size in hp["img_size"].domain.values:
                for colour in hp["colour"].domain.values:
                    for batch_size in hp["batch_size"].domain.values:
                        for lr in hp["lr"].domain.values:
                            for dropout in hp["dropout"].domain.values:
                                for relu_grad in hp["relu_grad"].domain.values:
                                    hparams = {hp["model"]: model,
                                               hp["optimizer"]: optimizer,
                                               hp["img_size"]: img_size,
                                               hp["colour"]: colour,
                                               hp["batch_size"]: batch_size,
                                               hp["lr"]: lr,
                                               hp["dropout"]: dropout,
                                               hp["relu_grad"]: relu_grad}
                                    dir_dict = directories(trial_id=trial_id, run_num=run_num, img_size=img_size, colour=colour)
                                    check_create_dataset(img_size=img_size, colour=colour, dir_dict=dir_dict)
                                    with open(dir_dict["trial_config"], "a") as f:
                                        [print(f"{key.name}: {hparams[key]}", file=f) for key in hparams.keys()]
                                    training(args=args, hparams=hparams, dir_dict=dir_dict)
                                    run_num += 1
