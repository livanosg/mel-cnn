import os

COLUMNS = ["dataset_id", "image", "image_type", "sex", "age_approx", "anatom_site_general", "class"]
MAPPER = {"image_type": {"clinic": 0, "derm": 1},
          "sex": {"m": 0, "male": 0,
                  "f": 1, "female": 1,
                  -10: -1},
          "age_approx": {-10: -1, 0: 0, 10: 1, 20: 2, 30: 3, 40: 4,
                         50: 5, 60: 6, 70: 7, 80: 8, 90: 9},
          "anatom_site_general": {'abdomen': 0, 'back': 0, 'chest': 0, 'anterior torso': 0,
                                  'posterior torso': 0, 'lateral torso': 0, 'torso': 0,
                                  'upper extremity': 1, 'upper_extremity': 1, 'upper limbs': 1,
                                  'head/neck': 2, 'head neck': 2,
                                  'lower extremity': 3, 'lower_extremity': 3, 'lower limbs': 3, 'buttocks': 3,
                                  'acral': 4, 'palms/soles': 4,
                                  'genital areas': 5, 'oral/genital': 5,
                                  -10: -1},
          "diagnosis": {'Common Nevus': 0, 'NV': 0, 'naevus': 0, 'ML': 0,
                        'blue nevus': 0, 'clark nevus': 0, 'combined nevus': 0, 'congenital nevus': 0,
                        'dermal nevus': 0, 'recurrent nevus': 0, 'reed or spitz nevus': 0,
                        'MEL': 1, 'melanoma': 1, 'Melanoma': 1, 'melanoma (in situ)': 1,
                        'melanoma (less than 0.76 mm)': 1,
                        'melanoma (0.76 to 1.5 mm)': 1, 'melanoma (more than 1.5 mm)': 1, 'melanoma metastasis': 1,
                        'melanosis': 2, 'miscellaneous': 2, 'vascular lesion': 2, 'seborrheic keratosis': 2, 'DF': 2,
                        'PYO': 2, 'SK': 2, 'VASC': 2, 'BKL': 2, 'dermatofibroma': 2, 'lentigo': 2,
                        'basal cell carcinoma': 3, 'BCC': 3, 'IEC': 3, 'SCC': 3,
                        'Atypical Nevus': 4, 'ANV': 4, 'AK': 4,
                        "unknown": 5
                        }
          }

BEN_MAL_MAPPER = {"class": {0: 0, 2: 0, 5: 0, 1: 1, 3: 1, 4: 2}}  # Group 0: NV, NNV, unknown | 1: MEL, NMC | 2: SUS
NEV_MEL_OTHER_MAPPER = {"class": {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2}}  # Group 0: NV, | 1: MEL | 2: NNV, NMC, SUS, unknown
IMAGE_FOLDER = "proc_{}_{}"


def directories(trial_id, mode, run_num, img_size, colour):
    dir_dict = {"main": os.path.dirname(os.path.abspath(__file__))}
    trial = f"{trial_id}-run-{str(run_num).zfill(4)}"
    dir_dict["logs"] = os.path.join(dir_dict["main"], "logs", f"{trial}_{mode}")
    dir_dict["trial"] = os.path.join(dir_dict["main"], "trials", f"{trial}_{mode}")
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
