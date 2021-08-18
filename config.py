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
DATA_MAP = {'image_type': {'clinic': 0,
                           'derm': 1,
                           },
          'sex': {'m': 0, 'male': 0,
                  'f': 1, 'female': 1,
                  np.nan: -1},
          'age_approx': {np.nan: -1, 0: 0, 10: 1, 20: 2, 30: 3, 40: 4,
                         50: 5, 60: 6, 70: 7, 80: 8, 90: 9},
            #  0: Torso | 1: Upper extremity | 2: Head and Neck | 3: Lower Extremity | 4: palms/soles | 5: Genital and oral
            'location': {'abdomen': 0, 'back': 0, 'chest': 0, 'anterior torso': 0, 'CHEST': 0, 'BACK': 0,
                                  'posterior torso': 0, 'lateral_torso': 0, 'torso': 0, 'ABDOMEN': 0,
                                  'upper extremity': 1, 'upper_extremity': 1, 'upper limbs': 1, 'ARM': 1, 'HAND': 1, 'FOREARM': 1,
                                  'head/neck': 2, 'head neck': 2, 'NECK': 2, 'FACE': 2, 'NOSE': 2, 'SCALP': 2, 'EAR': 2,
                                  'lower extremity': 3, 'lower_extremity': 3, 'lower limbs': 3, 'buttocks': 3, 'THIGH': 3, 'FOOT': 3,
                                  'acral': 4, 'palms/soles': 4,
                                  'genital areas': 5, 'oral/genital': 5, 'LIP': 5,
                                  np.nan: -1},
            #  0: Nevus | 1: Non-Nevus benign | 2: Suspicious | 3: Non-Melanocytic Carcinoma | 4: Melanoma
            'class':
              {'NV': 0, 'nevus': 0, 'clark nevus': 0, 'reed or spitz nevus': 0, 'naevus': 0, 'Common Nevus': 0, 'dermal nevus': 0,
               'blue nevus': 0, 'congenital nevus': 0, 'recurrent nevus': 0, 'combined nevus': 0, 'ML': 0, 'NEV': 0,

               'BKL': 1, 'DF': 1, 'VASC': 1, 'seborrheic keratosis': 1, 'lentigo NOS': 1, 'lichenoid keratosis': 1,
               'solar lentigo': 1, 'cafe-au-lait macule': 1, 'dermatofibroma': 1, 'lentigo': 1, 'melanosis': 1, 'vascular lesion': 1,
               'miscellaneous': 1, 'SK': 1, 'PYO': 1, 'SEK': 1, 'NNV': 1,

               'AKIEC': 2, 'AK': 2, 'SUS': 2, 'ANV': 2, 'atypical melanocytic proliferation': 2, 'Atypical Nevus': 2,
               'ACK': 2,

               'BCC': 3, 'SCC': 3, 'basal cell carcinoma': 3, 'IEC': 3, 'NMC': 3,


               'MEL': 4, 'melanoma': 4, 'melanoma (less than 0.76 mm)': 4, 'melanoma (0.76 to 1.5 mm)': 4,
               'melanoma (more than 1.5 mm)': 4, 'melanoma (in situ)': 4, 'melanoma metastasis': 4, 'Melanoma': 4,

               'unknown': 5
               }
            }

BEN_MAL_MAP = {'class': {0: 0, 1: 0, 2: 0, 5: 0,  # Group 0: NV, NNV, SUS, unknown | 1: MEL, NMC
                         3: 1, 4: 1}
               }
NEV_MEL_MAP = {'class': {0: 0,  # Group 0: NV, | 1: MEL | 2: NNV, NMC, SUS, unknown
                         4: 1,
                         1: 2, 2: 2, 3: 2, 5: 2}}
CLASS_NAMES = {'ben_mal': ['BEN', 'MAL'],
               'nev_mel': ['NEV', 'MEL'],
               '5cls': ['NEV', 'NEV', 'SUS', 'NMC', 'MEL']}


def directories(args):
    trial_id = datetime.now().strftime('%d%m%y%H%M%S')
    dir_dict = {'data': DATA_DIR,
                'data_csv': {'train': TRAIN_CSV_PATH,
                             'val': VAL_CSV_PATH,
                             'test': TEST_CSV_PATH,
                             'isic20_test': ISIC_ORIG_TEST_PATH},
                'logs': os.path.join(LOGS_DIR, args['mode'], args['image_type'], trial_id),
                'trial': os.path.join(TRIALS_DIR, args['mode'], args['image_type'], trial_id)}
    try:
        dir_dict['logs'] = dir_dict['logs'] + f"-{os.environ['SLURMD_NODENAME']}"
        dir_dict['trial'] = dir_dict['trial'] + f"-{os.environ['SLURMD_NODENAME']}"
    except KeyError:
        pass
    os.makedirs(dir_dict['logs'], exist_ok=True)
    os.makedirs(dir_dict['trial'], exist_ok=True)
    dir_dict['hparams_logs'] = os.path.join(dir_dict['trial'], 'hparams_log.txt')
    dir_dict['save_path'] = os.path.join(dir_dict['trial'], 'models', 'best-model')  # + "{epoch:03d}"
    dir_dict['backup'] = os.path.join(dir_dict['trial'], 'backup')
    dir_dict['image_folder'] = os.path.join(MAIN_DIR, f"proc_{args['image_size']}_{args['colour']}", 'data')
    return dir_dict
