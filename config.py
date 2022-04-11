import csv
import os

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MAIN_DIR, 'data')
LOGS_DIR = os.path.join(MAIN_DIR, 'logs')
TRIALS_DIR = os.path.join(MAIN_DIR, 'trials')
MODELS_DIR = os.path.join(MAIN_DIR, 'models')

TRAIN_CSV_PATH = os.path.join(MAIN_DIR, 'data_train.csv')
VAL_CSV_PATH = os.path.join(MAIN_DIR, 'data_val.csv')
TEST_CSV_PATH = os.path.join(MAIN_DIR, 'data_test.csv')
ISIC18_VAL_TEST_PATH = os.path.join(MAIN_DIR, 'isic18_val_test.csv')
DERMOFIT_TEST_PATH = os.path.join(MAIN_DIR, 'dermofit_test.csv')
UP_TEST_PATH = os.path.join(MAIN_DIR, 'up_test.csv')
ISIC16_TEST_PATH = os.path.join(MAIN_DIR, 'isic16_test.csv')
ISIC17_TEST_PATH = os.path.join(MAIN_DIR, 'isic17_test.csv')
ISIC20_TEST_PATH = os.path.join(MAIN_DIR, 'isic20_test.csv')
MCLASS_CLINIC_TEST_PATH = os.path.join(MAIN_DIR, 'mclass_clinic_test.csv')
MCLASS_DERM_TEST_PATH = os.path.join(MAIN_DIR, 'mclass_derm_test.csv')

COLUMNS = ['dataset_id', 'patient_id', 'lesion_id', 'image', 'image_type', 'sex', 'age_approx', 'location', 'class']

IMAGE_TYPE = ['clinic', 'derm']
SEX = ['male', 'female']
AGE_APPROX = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90']
LOCATIONS = ['torso', 'upper_extr', 'head_neck', 'lower_extr', 'palms_soles', 'genit_oral']
CLASS_NAMES = ['NEV', 'NNV', 'SUS', 'NMC', 'MEL', 'UNK']
TASK_CLASSES = {'ben_mal': ['BEN', 'MAL'],
                '5cls': ['NEV', 'NNV', 'SUS', 'NMC', 'MEL'],
                'nev_mel': ['NEV', 'MEL']}

# 0: Torso | 1: Upper extremity | 2: Head and Neck | 3: Lower Extremity | 4: palms/soles | 5: Genital and oral
DATA_MAP = {'location': {'abdomen': LOCATIONS[0], 'back': LOCATIONS[0], 'chest': LOCATIONS[0],
                         'anterior torso': LOCATIONS[0], 'CHEST': LOCATIONS[0], 'BACK': LOCATIONS[0],  # 0: torso
                         'posterior torso': 'torso', 'lateral_torso': 'torso', 'torso': 'torso', 'ABDOMEN': 'torso',

                         'upper extremity': LOCATIONS[1], 'upper_extremity': LOCATIONS[1], 'upper limbs': LOCATIONS[1],
                         'ARM': LOCATIONS[1], 'HAND': LOCATIONS[1], 'FOREARM': LOCATIONS[1],

                         'head/neck': LOCATIONS[2], 'head neck': LOCATIONS[2], 'NECK': LOCATIONS[2],
                         'FACE': LOCATIONS[2], 'NOSE': LOCATIONS[2], 'SCALP': LOCATIONS[2], 'EAR': LOCATIONS[2],

                         'lower extremity': LOCATIONS[3], 'lower_extremity': LOCATIONS[3], 'lower limbs': LOCATIONS[3],
                         'buttocks': LOCATIONS[3], 'THIGH': LOCATIONS[3], 'FOOT': LOCATIONS[3],

                         'acral': LOCATIONS[4], 'palms/soles': LOCATIONS[4],

                         'genital areas': LOCATIONS[5], 'oral/genital': LOCATIONS[5], 'LIP': LOCATIONS[5]},

            #  0: Nevus | 1: Non-Nevus benign | 2: Suspicious | 3: Non-Melanocytic Carcinoma | 4: Melanoma | 5: Unknown
            'class': {'NV': CLASS_NAMES[0], 'nevus': CLASS_NAMES[0], 'clark nevus': CLASS_NAMES[0],
                      'reed or spitz nevus': CLASS_NAMES[0], 'naevus': CLASS_NAMES[0], 'Common Nevus': CLASS_NAMES[0],
                      'dermal nevus': CLASS_NAMES[0], 'blue nevus': CLASS_NAMES[0], 'congenital nevus': CLASS_NAMES[0],
                      'recurrent nevus': CLASS_NAMES[0], 'combined nevus': CLASS_NAMES[0], 'ML': CLASS_NAMES[0],
                      'NEV': CLASS_NAMES[0],

                      'BKL': CLASS_NAMES[1], 'DF': CLASS_NAMES[1], 'VASC': CLASS_NAMES[1],
                      'seborrheic keratosis': CLASS_NAMES[1], 'lentigo NOS': CLASS_NAMES[1],
                      'lichenoid keratosis': CLASS_NAMES[1], 'solar lentigo': CLASS_NAMES[1],
                      'cafe-au-lait macule': CLASS_NAMES[1], 'dermatofibroma': CLASS_NAMES[1],
                      'lentigo': CLASS_NAMES[1], 'melanosis': CLASS_NAMES[1], 'PYO': CLASS_NAMES[1],
                      'miscellaneous': CLASS_NAMES[1], 'SK': CLASS_NAMES[1], 'NNV': CLASS_NAMES[1],
                      'SEK': CLASS_NAMES[1], 'vascular lesion': CLASS_NAMES[1],

                      'AKIEC': CLASS_NAMES[2], 'AK': CLASS_NAMES[2], 'SUS': CLASS_NAMES[2],
                      'ANV': CLASS_NAMES[2], 'atypical melanocytic proliferation': CLASS_NAMES[2],
                      'Atypical Nevus': CLASS_NAMES[2], 'ACK': CLASS_NAMES[2],

                      'BCC': CLASS_NAMES[3], 'SCC': CLASS_NAMES[3], 'basal cell carcinoma': CLASS_NAMES[3],
                      'IEC': CLASS_NAMES[3], 'NMC': CLASS_NAMES[3],

                      'MEL': CLASS_NAMES[4], 'melanoma': CLASS_NAMES[4], 'melanoma (0.76 to 1.5 mm)': CLASS_NAMES[4],
                      'melanoma (less than 0.76 mm)': CLASS_NAMES[4], 'melanoma (in situ)': CLASS_NAMES[4],
                      'melanoma (more than 1.5 mm)': CLASS_NAMES[4], 'Melanoma': CLASS_NAMES[4],
                      'melanoma metastasis': CLASS_NAMES[4],

                      'unknown': CLASS_NAMES[5]}
            }
# 0: NV, NNV, SUS, UNK | 1: MEL, NMC
BEN_MAL_MAP = {'class': {'BEN': 'BEN', CLASS_NAMES[0]: 'BEN', CLASS_NAMES[1]: 'BEN', CLASS_NAMES[2]: 'BEN', CLASS_NAMES[5]: 'BEN',
                         'MAL': 'MAL', CLASS_NAMES[3]: 'MAL', CLASS_NAMES[4]: 'MAL'}
               }


def dir_dict(args: dict):
    """ Set paths for each run with unique path for example:
          logs/5cls/both/130921080509
        trials/5cls/both/130921080509
        models/5cls/both/130921080509/model
        MAIN_PATH --- LOGS_DIR: Tensorboard logs
                  --- TRIALS_DIR: Hparams and results
                  --- MODELS_DIR: saved model
              """

    exp_path = os.path.join(args['task'], args['image_type'], args['trial_id'])
    directories = {'data': DATA_DIR,
                   'data_csv': {'train': TRAIN_CSV_PATH,
                                'val': VAL_CSV_PATH,
                                'test': TEST_CSV_PATH,
                                'isic20_test': ISIC20_TEST_PATH}}
    if not args['load_model']:
        directories['logs'] = os.path.join(LOGS_DIR, exp_path)
        directories['trial'] = os.path.join(TRIALS_DIR, exp_path)
        directories['save_path'] = os.path.join(MODELS_DIR, exp_path)
        if os.getenv('SLURMD_NODENAME'):  # Append node name if training on HPC with SLURM.
            directories['logs'] = '-'.join([directories['logs'], os.getenv('SLURMD_NODENAME')])
            directories['trial'] = '-'.join([directories['trial'], os.getenv('SLURMD_NODENAME')])
            directories['save_path'] = '-'.join([directories['save_path'], os.getenv('SLURMD_NODENAME')])
    else:
        # Use full path to load model
        directories['logs'] = args['load_model'].replace(MODELS_DIR, LOGS_DIR)
        directories['trial'] = args['load_model'].replace(MODELS_DIR, TRIALS_DIR)
        directories['save_path'] = args['load_model']
        if args['fine'] and (not args['load_model'].endswith('_fine')):
            directories['logs'] = directories['logs'] + '_fine'
            directories['trial'] = directories['trial'] + '_fine'
            directories['save_path'] = directories['save_path'] + '_fine'

    directories['hparams_logs'] = os.path.join(MAIN_DIR, 'hparams_log.csv')
    directories['model_summary'] = os.path.join(directories['trial'], 'model_summary.txt')
    directories['train_logs'] = os.path.join(directories['trial'], 'train_logs.csv')
    directories['data_folder'] = os.path.join(MAIN_DIR, f"proc_{args['image_size']}", 'data')
    if not args['test']:
        os.makedirs(directories['logs'], exist_ok=True)
        os.makedirs(directories['trial'], exist_ok=True)
    return directories


def log_params(args):
    if os.path.exists(args['dir_dict']['hparams_logs']):
        aw = 'a'
    else:
        aw = 'w'
    with open(args['dir_dict']['hparams_logs'], aw) as f:
        fieldnames = ['trial_id', 'task', 'num_classes', 'class_names', 'image_type', 'image_size', 'input_shape',
                      'no_clinical_data', 'clinic_val', 'no_image_type', 'conv_layers', 'dense_layers', 'merge_layers',
                      'l1_reg', 'l2_reg', 'loss_fn', 'loss_frac', 'weighted_samples', 'weighted_loss',
                      'dataset_frac', 'pretrained', 'batch_size', 'learning_rate', 'optimizer', 'activation',
                      'dropout', 'epochs', 'test', 'load_model', 'fine', 'gpus', 'strategy']
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='', extrasaction='ignore')
        if aw == 'w':
            writer.writeheader()
        writer.writerows([args])
