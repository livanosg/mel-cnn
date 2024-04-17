import os
import csv
import argparse
from datetime import datetime

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MAIN_DIR, 'data')
LOGS_DIR = os.path.join(MAIN_DIR, 'logs')
TRIALS_DIR = os.path.join(MAIN_DIR, 'trials')
MODELS_DIR = os.path.join(MAIN_DIR, 'models')
INFO_DIR = os.path.join(MAIN_DIR, 'data_info')
HPARAMS_FILE = os.path.join(MAIN_DIR, 'hparams_log.csv')
PROC_DATA_DIR = ''


def log_params(args, dirs):
    exists = os.path.exists(path=dirs['hparams_log'])
    with open(dirs['hparams_log'], 'a' if exists else 'w') as f:
        fieldnames = args.keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerows([args])
