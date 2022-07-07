import os
from settings import MAIN_DIR

args = {'no_image_type': False,
        'task': 'ben_mal',
        'no_clinical_data': False,
        'weighted_samples': False,
        'weighted_loss': False,
        'dataset_frac': 1.,
        'image_type': 'both'
        }
dirs = {'proc_img_folder': os.path.join(MAIN_DIR, f"proc_{224}")}

