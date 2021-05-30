import os
from datetime import datetime

COLUMNS = ["dataset_id", "image", "image_type", "sex", "age_approx", "anatom_site_general", "class"]
MAPPER = {"image_type": {"clinic": 0, "derm": 1},
          "sex": {"m": 0, "f": 1},
          "age_approx": {"0": 0, "10": 1, "20": 2, "30": 3, "40": 4,
                         "50": 5, "60": 6, "70": 7, "80": 8, "90": 9},
          "anatom_site_general": {"head/neck": 0, "torso": 1, "lateral_torso": 2,
                                  "upper_extremity": 3, "lower_extremity": 4,
                                  "palms/soles": 5, "oral/genital": 6},
          "class": {"NV": 0, "NNV": 1, "MEL": 2, "NMC": 3, "SUS": 4}}
IMAGE_FOLDER = "proc_{}_{}"


def directories(trial_id, run_num, img_size, colour):
    dir_dict = {"main": os.path.dirname(os.path.abspath(__file__))}
    trial = f"{trial_id}-run-{str(run_num).zfill(4)}"
    dir_dict["logs"] = os.path.join(dir_dict["main"], "logs", trial)
    dir_dict["trial"] = os.path.join(dir_dict["main"], "trials", trial)
    try:
        dir_dict["logs"] = f"{os.environ['SLURMD_NODENAME']}-" + dir_dict["logs"]
        dir_dict["trial"] = f"{os.environ['SLURMD_NODENAME']}-" + dir_dict["trial"]
    except KeyError:
        pass
    os.makedirs(dir_dict["logs"], exist_ok=True)
    os.makedirs(dir_dict["trial"], exist_ok=True)
    dir_dict["trial_config"] = os.path.join(dir_dict["trial"], "log_conf.txt")
    dir_dict["save_path"] = os.path.join(dir_dict["trial"], "models", "best-model")  # + "{epoch:03d}"
    dir_dict["backup"] = os.path.join(dir_dict["trial"], "backup")
    dir_dict["image_folder"] = os.path.join(dir_dict["main"], IMAGE_FOLDER.format(str(img_size), colour))
    return dir_dict
