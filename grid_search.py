from datetime import datetime
from config import directories
from prep_dataset import check_create_dataset
from train_model import training


def grid(args):
    run_num = 0
    trial_id = datetime.now().strftime('%d%m%y%H%M%S')
    dir_dict = directories(trial_id=trial_id, run_num=run_num, args=args)
    args["dir_dict"] = dir_dict
    check_create_dataset(args=args)
    with open(args["dir_dict"]["hparams_logs"], "a") as f:
        [f.write(f"{key.capitalize()}: {args[key]}\n") for key in args.keys() if key not in ("dir_dict", "hparams")]
        f.write("Directories\n")
        [f.write(f"{key.capitalize()}: {args['dir_dict'][key]}\n") for key in args["dir_dict"].keys()]
    training(args=args)
    run_num += 1
