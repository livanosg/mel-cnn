import os
from datetime import datetime
import numpy as np

NP_RNG = np.random.default_rng(1312)
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MAIN_DIR, 'data')
LOGS_DIR = os.path.join(MAIN_DIR, 'logs')
TRIALS_DIR = os.path.join(MAIN_DIR, 'trials')

TRAIN_CSV_PATH = os.path.join(MAIN_DIR, 'data_train.csv')
VAL_CSV_PATH = os.path.join(MAIN_DIR, 'data_val.csv')
TEST_CSV_PATH = os.path.join(MAIN_DIR, 'data_test.csv')
ISIC_ORIG_TEST_PATH = os.path.join(MAIN_DIR, 'isic20_test.csv')

COLUMNS = ['dataset_id', 'patient_id', 'lesion_id', 'image', 'image_type', 'sex', 'age_approx', 'location', 'class']

IMAGE_TYPE = ['clinic', 'derm']
SEX = ['male', 'female']
AGE_APPROX = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
LOCATIONS = ['torso', 'upper_extr', 'head_neck', 'lower_extr', 'palms_soles', 'genit_oral']
CLASS_NAMES = ['NEV', 'NNV', 'SUS', 'NMC', 'MEL', 'UNK']
TASK_CLASSES = {'ben_mal': ['BEN', 'MAL'],
                '5cls': CLASS_NAMES[:-1],
                'nev_mel': [CLASS_NAMES[idx] for idx in [0, 4]]}

#  0: Torso | 1: Upper extremity | 2: Head and Neck | 3: Lower Extremity | 4: palms/soles | 5: Genital and oral
DATA_MAP = {'location': {'abdomen': LOCATIONS[0], 'back': LOCATIONS[0], 'chest': LOCATIONS[0], 'anterior torso': LOCATIONS[0],
                         'CHEST': LOCATIONS[0], 'BACK': LOCATIONS[0],  # 0: torso
                         'posterior torso': 'torso', 'lateral_torso': 'torso', 'torso': 'torso', 'ABDOMEN': 'torso',
                         # 1: Upper extremity
                         'upper extremity': LOCATIONS[1], 'upper_extremity': LOCATIONS[1], 'upper limbs': LOCATIONS[1],
                         'ARM': LOCATIONS[1], 'HAND': LOCATIONS[1], 'FOREARM': LOCATIONS[1],
                         # 2: Head and Neck
                         'head/neck': LOCATIONS[2], 'head neck': LOCATIONS[2], 'NECK': LOCATIONS[2], 'FACE': LOCATIONS[2],
                         'NOSE': LOCATIONS[2], 'SCALP': LOCATIONS[2], 'EAR': LOCATIONS[2],
                         # 3: Lower Extremity
                         'lower extremity': LOCATIONS[3], 'lower_extremity': LOCATIONS[3], 'lower limbs': LOCATIONS[3],
                         'buttocks': LOCATIONS[3], 'THIGH': LOCATIONS[3], 'FOOT': LOCATIONS[3],
                         # 4: palms/soles
                         'acral': LOCATIONS[4], 'palms/soles': LOCATIONS[4],
                         # 5: Genital and oral
                         'genital areas': LOCATIONS[5], 'oral/genital': LOCATIONS[5], 'LIP': LOCATIONS[5]},
            #  0: Nevus
            'class': {'NV': CLASS_NAMES[0], 'nevus': CLASS_NAMES[0], 'clark nevus': CLASS_NAMES[0], 'reed or spitz nevus': CLASS_NAMES[0],
                      'naevus': CLASS_NAMES[0], 'Common Nevus': CLASS_NAMES[0], 'dermal nevus': CLASS_NAMES[0],
                      'blue nevus': CLASS_NAMES[0], 'congenital nevus': CLASS_NAMES[0], 'recurrent nevus': CLASS_NAMES[0],
                      'combined nevus': CLASS_NAMES[0], 'ML': CLASS_NAMES[0], 'NEV': CLASS_NAMES[0],
                      # 1: Non-Nevus benign
                      'BKL': CLASS_NAMES[1], 'DF': CLASS_NAMES[1], 'VASC': CLASS_NAMES[1], 'seborrheic keratosis': CLASS_NAMES[1], 'lentigo NOS': CLASS_NAMES[1],
                      'lichenoid keratosis': CLASS_NAMES[1], 'solar lentigo': CLASS_NAMES[1], 'cafe-au-lait macule': CLASS_NAMES[1],
                      'dermatofibroma': CLASS_NAMES[1], 'lentigo': CLASS_NAMES[1], 'melanosis': CLASS_NAMES[1], 'vascular lesion': CLASS_NAMES[1],
                      'miscellaneous': CLASS_NAMES[1], 'SK': CLASS_NAMES[1], 'PYO': CLASS_NAMES[1], 'SEK': CLASS_NAMES[1], 'NNV': CLASS_NAMES[1],
                      # 2: Suspicious
                      'AKIEC': CLASS_NAMES[2], 'AK': CLASS_NAMES[2], 'SUS': CLASS_NAMES[2], 'ANV': CLASS_NAMES[2],
                      'atypical melanocytic proliferation': CLASS_NAMES[2], 'Atypical Nevus': CLASS_NAMES[2], 'ACK': CLASS_NAMES[2],
                      # 3: Non-Melanocytic Carcinoma
                      'BCC': CLASS_NAMES[3], 'SCC': CLASS_NAMES[3], 'basal cell carcinoma': CLASS_NAMES[3], 'IEC': CLASS_NAMES[3], 'NMC': CLASS_NAMES[3],
                      # 4: Melanoma
                      'MEL': CLASS_NAMES[4], 'melanoma': CLASS_NAMES[4], 'melanoma (less than 0.76 mm)': CLASS_NAMES[4],
                      'melanoma (0.76 to 1.5 mm)': CLASS_NAMES[4], 'melanoma (more than 1.5 mm)': CLASS_NAMES[4],
                      'melanoma (in situ)': CLASS_NAMES[4], 'melanoma metastasis': CLASS_NAMES[4], 'Melanoma': CLASS_NAMES[4],
                      # 5: Unknown
                      'unknown': CLASS_NAMES[5]}
            }

BEN_MAL_MAP = {'class': {CLASS_NAMES[0]: 'BEN', CLASS_NAMES[1]: 'BEN', CLASS_NAMES[2]: 'BEN', CLASS_NAMES[5]: 'BEN',  # Group 0: NV, NNV, SUS, UNK
                         CLASS_NAMES[3]: 'MAL', CLASS_NAMES[4]: 'MAL'}  # | 1: MEL, NMC
               }


def dir_dict(args: dict):
    trial_id = datetime.now().strftime('%d%m%y%H%M%S')
    directories = {'data': DATA_DIR,
                   'data_csv': {'train': TRAIN_CSV_PATH,
                                'val': VAL_CSV_PATH,
                                'test': TEST_CSV_PATH,
                                'isic20_test': ISIC_ORIG_TEST_PATH},
                   'logs': os.path.join(LOGS_DIR, args['task'], args['image_type'], trial_id),
                   'trial': os.path.join(TRIALS_DIR, args['task'], args['image_type'], trial_id)}  # type: dict
    try:
        directories['logs'] = directories['logs'] + f"-{os.environ['SLURMD_NODENAME']}"
        directories['trial'] = directories['trial'] + f"-{os.environ['SLURMD_NODENAME']}"
    except KeyError:
        pass
    directories['hparams_logs'] = os.path.join(directories['trial'], 'hparams_log.txt')
    directories['model_path'] = os.path.join(directories['trial'], 'model')  # + "{epoch:03d}"
    directories['backup'] = os.path.join(directories['trial'], 'backup')
    directories['image_folder'] = os.path.join(MAIN_DIR, f"proc_{args['image_size']}_{args['colour']}", 'data')
    return directories
