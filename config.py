import os

COLUMNS = ["dataset_id", "image", "image_type", "sex", "age_approx", "anatom_site_general", "class"]
MAPPER = {"image_type": {"clinic": 0,
                         "derm": 1,
                         },
          "sex": {"m": 0, "male": 0,
                  "f": 1, "female": 1,
                  -10: -1},
          "age_approx": {-10: -1, 0: 0, 10: 1, 20: 2, 30: 3, 40: 4,
                         50: 5, 60: 6, 70: 7, 80: 8, 90: 9},
          "anatom_site_general": {'abdomen': 0, 'back': 0, 'chest': 0, 'anterior torso': 0,
                                  'posterior torso': 0, 'lateral_torso': 0, 'torso': 0,
                                  'upper extremity': 1, 'upper_extremity': 1, 'upper limbs': 1,
                                  'head/neck': 2, 'head neck': 2,
                                  'lower extremity': 3, 'lower_extremity': 3, 'lower limbs': 3, 'buttocks': 3,
                                  'acral': 4, 'palms/soles': 4,
                                  'genital areas': 5, 'oral/genital': 5,
                                  -10: -1},
          #  0: Nevus | 1: Melanoma | 2: Non-Nevus benign | 3: Non-Melanocytic Carcinoma | 4: Suspicious
          "class":
              {'Common Nevus': 0, 'NV': 0, 'nevus': 0, 'naevus': 0, 'ML': 0, 'blue nevus': 0, 'clark nevus': 0,
               'combined nevus': 0, 'congenital nevus': 0, 'dermal nevus': 0, 'recurrent nevus': 0,
               'reed or spitz nevus': 0,
               'MEL': 1, 'melanoma': 1, 'Melanoma': 1, 'melanoma (in situ)': 1, 'melanoma (less than 0.76 mm)': 1,
               'melanoma (0.76 to 1.5 mm)': 1, 'melanoma (more than 1.5 mm)': 1, 'melanoma metastasis': 1,
               'NNV': 2, 'cafe-au-lait macule': 2, 'melanosis': 2, 'miscellaneous': 2, 'vascular lesion': 2, 'seborrheic keratosis': 2,
               'DF': 2, 'PYO': 2, 'SK': 2, 'VASC': 2, 'BKL': 2, 'dermatofibroma': 2, 'lentigo': 2, 'lentigo NOS': 2,
               'solar lentigo': 2, 'lichenoid keratosis': 2,
               'basal cell carcinoma': 3, 'BCC': 3, 'AKIEC': 3, 'IEC': 3, 'SCC': 3, 'NMC': 3,
               'Atypical Nevus': 4, 'ANV': 4, 'AK': 4, 'SUS': 4, 'atypical melanocytic proliferation': 4,
               "unknown": 5
               }
          }

BEN_MAL_MAPPER = {"class": {0: 0, 2: 0, 5: 0,  # Group 0: NV, NNV, unknown | 1: MEL, NMC | 2: SUS
                            1: 1, 3: 1,
                            4: 2}
                  }
NEV_MEL_OTHER_MAPPER = {"class": {0: 0,  # Group 0: NV, | 1: MEL | 2: NNV, NMC, SUS, unknown
                                  1: 1,
                                  2: 2, 3: 2, 4: 2, 5: 2}}
CLASS_NAMES = {"ben_mal": ["Benign", "Malignant"],
               "nev_mel": ["Nevus", "Melanoma"],
               "5cls": ["Nevus", "Melanoma", "Non-Nevus benign", "Non-Melanocytic Carcinoma", "Suspicious benign"]}

TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
TEST_CSV = "test.csv"


def directories(trial_id, run_num, args):
    trial = f"{trial_id}-run-{str(run_num).zfill(4)}"
    dir_dict = {"main": os.path.dirname(os.path.abspath(__file__))}
    dir_dict["data"] = os.path.join(dir_dict["main"], "data")
    dir_dict["data_csv"] = {"train": os.path.join(dir_dict["main"], TRAIN_CSV),
                            "val": os.path.join(dir_dict["main"], VAL_CSV),
                            "test": os.path.join(dir_dict["main"], TEST_CSV)}
    dir_dict["logs"] = os.path.join(dir_dict["main"], "logs", args['mode'], args['image_type'], trial)
    dir_dict["trial"] = os.path.join(dir_dict["main"], "trials", args['mode'], args['image_type'], trial)
    try:
        dir_dict["logs"] = dir_dict["logs"] + f"-{os.environ['SLURMD_NODENAME']}"
        dir_dict["trial"] = dir_dict["trial"] + f"-{os.environ['SLURMD_NODENAME']}"
    except KeyError:
        pass
    os.makedirs(dir_dict["logs"], exist_ok=True)
    os.makedirs(dir_dict["trial"], exist_ok=True)
    dir_dict["hparams_logs"] = os.path.join(dir_dict["trial"], "hparams_log.txt")
    dir_dict["save_path"] = os.path.join(dir_dict["trial"], "models", "best-model")  # + "{epoch:03d}"
    dir_dict["backup"] = os.path.join(dir_dict["trial"], "backup")
    dir_dict["image_folder"] = os.path.join(dir_dict["main"], f"proc_{args['image_size']}_{args['colour']}")
    return dir_dict


if __name__ == '__main__':
    test_args = {"mode": "nev_mel",
                 "dataset_frac": 1,
                 "image_type": "both",
                 "image_size": 100}

    a = directories(trial_id=1, run_num=0, args=test_args)
    print(a["main"])
