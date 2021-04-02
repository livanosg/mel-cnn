CLASSES_DICT = {'NV': 0, "NNV": 1, "MEL": 2, "NMC": 3, "SUS": 4}
ANATOM_SITE_DICT = {'head neck': 0, 'torso': 1, 'lateral torso': 2, 'upper extremity': 3,
                    'lower extremity': 4, 'palms soles': 5, 'oral genital': 6, 'nan': 7}
IMAGE_TYPE_DICT = {'clinic': 0, 'derm': 1}
SEX_DICT = {'m': 0, 'f': 1, 'nan': 2}


COLUMNS = ['dataset_id', 'image', 'image_type', 'sex', 'age_approx', 'anatom_site_general', 'class']
BUFFER_SIZE = 1000
