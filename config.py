COLUMNS = ['dataset_id', 'image', 'image_type', 'sex', 'age_approx', 'anatom_site_general', 'class']
IMAGE_TYPE_MAP = {'clinic': 0, 'derm': 1}
SEX_MAP = {'m': 0, 'f': 1}
AGE_APPROX_MAP = {'0': 0, '10': 1, '20': 2, '30': 3, '40': 4,
                  '50': 5, '60': 6, '70': 7, '80': 8, '90': 9}
ANATOM_SITE_MAP = {'head/neck': 0, 'torso': 1, 'lateral_torso': 2,
                   'upper_extremity': 3, 'lower_extremity': 4,
                   'palms/soles': 5, 'oral/genital': 6}
CLASSES_DICT = {'NV': 0, "NNV": 1, "MEL": 2, "NMC": 3, "SUS": 4}

MAPPER = {'image_type': IMAGE_TYPE_MAP,
          'sex': SEX_MAP,
          'age_approx': AGE_APPROX_MAP,
          'anatom_site_general': ANATOM_SITE_MAP,
          'class': CLASSES_DICT}
