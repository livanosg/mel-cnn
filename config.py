COLUMNS = ['dataset_id', 'image', 'image_type', 'sex', 'age_approx', 'anatom_site_general', 'class']
CLASSES_DICT = {'NV': 0, "NNV": 1, "MEL": 2, "NMC": 3, "SUS": 4}
ANATOM_SITE_DICT = {'head/neck': 0,
                    'torso': 1,
                    'lateral_torso': 2,
                    'upper_extremity': 3,
                    'lower_extremity': 4,
                    'palms/soles': 5,
                    'oral/genital': 6}
IMAGE_TYPE_DICT = {'clinic': 0, 'derm': 1}
SEX_DICT = {'m': 0, 'f': 1}
