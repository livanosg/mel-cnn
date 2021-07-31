from datetime import datetime
from config import directories
from prep_dataset import check_create_dataset
from hyperparameters import hp_dict
from train_model import training


def grind(args):
    hp = hp_dict(args=args)
    run_num = 0
    trial_id = datetime.now().strftime('%d%m%y%H%M%S')
    for model in hp["model"].domain.values:
        for optimizer in hp["optimizer"].domain.values:
            for image_size in hp["image_size"].domain.values:
                for image_type in hp["image_type"].domain.values:
                    for colour in hp["colour"].domain.values:
                        for batch_size in hp["batch_size"].domain.values:
                            for lr in hp["lr"].domain.values:
                                for dropout in hp["dropout"].domain.values:
                                    for relu_grad in hp["relu_grad"].domain.values:
                                        hparams = {hp["model"]: model,
                                                   hp["optimizer"]: optimizer,
                                                   hp["image_size"]: image_size,
                                                   hp["image_type"]: image_type,
                                                   hp["colour"]: colour,
                                                   hp["batch_size"]: batch_size,
                                                   hp["lr"]: lr,
                                                   hp["dropout"]: dropout,
                                                   hp["relu_grad"]: relu_grad}
                                        trial_args = dict((key.name, hparams[key]) for key in hparams.keys())
                                        trial_args["dataset_frac"] = args["dataset_frac"]
                                        trial_args["epochs"] = args["epochs"]
                                        trial_args["early_stop"] = args["early_stop"]
                                        trial_args["nodes"] = args["nodes"]
                                        trial_args["layers"] = args["layers"]
                                        trial_args["mode"] = args["mode"]
                                        trial_args["verbose"] = args["verbose"]
                                        trial_args["hparams"] = hparams
                                        dir_dict = directories(trial_id=trial_id, run_num=run_num, args=trial_args)
                                        trial_args["dir_dict"] = dir_dict
                                        check_create_dataset(args=trial_args)
                                        [print(f"{key.capitalize()}: {trial_args['dir_dict'][key]}") for key in trial_args["dir_dict"].keys()]
                                        [print(f"{key.capitalize()}: {trial_args[key]}") for key in trial_args.keys() if key not in ("dir_dict", "hparams")]
                                        with open(trial_args["dir_dict"]["hparams_logs"], "a") as f:
                                            [f.write(f"{key.capitalize()}: {trial_args[key]}\n") for key in trial_args.keys() if key not in ("dir_dict", "hparams")]
                                            f.write("Directories\n")
                                            [f.write(f"{key.capitalize()}: {trial_args['dir_dict'][key]}\n") for key in trial_args["dir_dict"].keys()]
                                        training(args=trial_args)
                                        run_num += 1
