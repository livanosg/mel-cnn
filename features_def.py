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
BEN_MAL_MAP = {
    'class': {'BEN': 'BEN', CLASS_NAMES[0]: 'BEN', CLASS_NAMES[1]: 'BEN', CLASS_NAMES[2]: 'BEN', CLASS_NAMES[5]: 'BEN',
              'MAL': 'MAL', CLASS_NAMES[3]: 'MAL', CLASS_NAMES[4]: 'MAL'}
    }
