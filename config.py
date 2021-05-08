import os
from datetime import datetime

from prep_dataset import check_create_dataset

COLUMNS = ["dataset_id", "image", "image_type", "sex", "age_approx", "anatom_site_general", "class"]
IMAGE_TYPE_MAP = {"clinic": 0, "derm": 1}
SEX_MAP = {"m": 0, "f": 1}
AGE_APPROX_MAP = {"0": 0, "10": 1, "20": 2, "30": 3, "40": 4,
                  "50": 5, "60": 6, "70": 7, "80": 8, "90": 9}
ANATOM_SITE_MAP = {"head/neck": 0, "torso": 1, "lateral_torso": 2,
                   "upper_extremity": 3, "lower_extremity": 4,
                   "palms/soles": 5, "oral/genital": 6}
CLASSES_DICT = {"NV": 0, "NNV": 1, "MEL": 2, "NMC": 3, "SUS": 4}

MAPPER = {"image_type": IMAGE_TYPE_MAP,
          "sex": SEX_MAP,
          "age_approx": AGE_APPROX_MAP,
          "anatom_site_general": ANATOM_SITE_MAP,
          "class": CLASSES_DICT}

IMAGE_FOLDER = "proc_{}_{}"


def directories(run_num, img_size, colour):
    dir_dict = {"main": os.path.dirname(os.path.abspath(__file__))}
    trial = f"run-{str(run_num).zfill(4)}-{datetime.now().strftime('%d%m%y%H%M%S')}"
    dir_dict["logs"] = os.path.join(dir_dict["main"], "logs", trial)
    dir_dict["trial"] = os.path.join(dir_dict["main"], "trials", trial)
    dir_dict["trial_config"] = os.path.join(dir_dict["trial"], "log_conf.txt")
    dir_dict["save_path"] = os.path.join(dir_dict["trial"], "models", "best-model")  # + "{epoch:03d}"
    dir_dict["backup"] = os.path.join(dir_dict["trial"], "backup")
    dir_dict["image_folder"] = os.path.join(dir_dict["main"], IMAGE_FOLDER.format(str(img_size), colour))
    try:
        dir_dict["logs"] += f"-{os.environ['SLURMD_NODENAME']}"
        dir_dict["trial"] += f"-{os.environ['SLURMD_NODENAME']}"
    except KeyError:
        pass
    os.makedirs(dir_dict["logs"], exist_ok=True)
    os.makedirs(dir_dict["trial"], exist_ok=True)
    return dir_dict


if __name__ == "__main__":
    check_create_dataset(directories(10, img_size=224, colour="grey")["image_folder"])
    check_create_dataset(directories(10, img_size=500, colour="grey")["image_folder"])
