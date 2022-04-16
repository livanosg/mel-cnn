import sys
import os
import csv
import argparse
from datetime import datetime

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
INIT_DATA_DIR = os.path.join(MAIN_DIR, 'data')
LOGS_DIR = os.path.join(MAIN_DIR, 'logs')
TRIALS_DIR = os.path.join(MAIN_DIR, 'trials')
MODELS_DIR = os.path.join(MAIN_DIR, 'models')
INFO_DIR = os.path.join(MAIN_DIR, 'data_info')
HPARAMS_FILE = os.path.join(MAIN_DIR, 'hparams_log.csv')
data_csv = {'train': os.path.join(MAIN_DIR, 'data_train.csv'),
            'validation': os.path.join(MAIN_DIR, 'data_val.csv'),
            'test': os.path.join(MAIN_DIR, 'data_test.csv'),
            'isic16_test': os.path.join(MAIN_DIR, 'isic16_test.csv'),
            'isic17_test': os.path.join(MAIN_DIR, 'isic17_test.csv'),
            'isic18_val_test': os.path.join(MAIN_DIR, 'isic18_val_test.csv'),
            'isic20_test': os.path.join(MAIN_DIR, 'isic20_test.csv'),
            'dermofit_test': os.path.join(MAIN_DIR, 'dermofit_test.csv'),
            'up_test': os.path.join(MAIN_DIR, 'up_test.csv'),
            'mclass_clinic_test': os.path.join(MAIN_DIR, 'mclass_clinic_test.csv'),
            'mclass_derm_test': os.path.join(MAIN_DIR, 'mclass_derm_test.csv')}


def parser():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--trial-id', '-id', type=str, default=datetime.now().strftime('%d%m%y%H%M%S'),
                             help='Select pretrained model.')
    args_parser.add_argument('--task', '-task', type=str, default='ben_mal', choices=['5cls', 'ben_mal', 'nev_mel'],
                             help='Select the type of model.')
    args_parser.add_argument('--image-type', '-it', type=str, default='both', choices=['derm', 'clinic', 'both'],
                             help='Select image type to use during training.')
    args_parser.add_argument('--image-size', '-is', type=int, default=224, help='Select image size.')
    args_parser.add_argument('--no-clinical-data', '-ncd', action='store_true', help='Train model only with images.')
    args_parser.add_argument('--no-image-type', '-nit', action='store_true',
                             help='Set to remove image type from training.')
    args_parser.add_argument('--clinic-val', '-cval', action='store_true', help='Run validation on clinical images only.')
    args_parser.add_argument('--conv_layers', '-clrs', type=int, default=32,
                             help='Select multiplier for number of nodes in inception layers.')
    args_parser.add_argument('--dense-layers', '-dlrs', type=int, default=16,
                             help='Select multiplier for number of nodes in dense layers.')
    args_parser.add_argument('--merge-layers', '-mlrs', type=int, default=32,
                             help='Select multiplier for number of nodes in merge layers.')
    args_parser.add_argument('--l1-reg', '-l1', type=float, default=0., help='L1 regularization.')
    args_parser.add_argument('--l2-reg', '-l2', type=float, default=0., help='L2 regularization.')
    args_parser.add_argument('--loss-fn', '-loss', type=str, default='cxe',
                             choices=['cxe', 'focal', 'perclass', 'wcxe', 'combined'], help='Select loss function.')
    args_parser.add_argument('--loss-frac', '-lossf', type=float, default=.5,
                             help='log_dice_loss ratio in custom loss.')
    args_parser.add_argument('--weighted-samples', '-ws', action='store_true',
                             help='Apply sample weights per image type.')
    args_parser.add_argument('--weighted-loss', '-wl', action='store_true', help='Apply class weights.')
    args_parser.add_argument('--dataset-frac', '-frac', type=float, default=1., help='Dataset fraction.')
    args_parser.add_argument('--pretrained', '-pt', type=str, default='effnet6',
                             choices=['incept', 'xept', 'effnet0', 'effnet1', 'effnet6'],
                             help='Select pretrained model.')
    args_parser.add_argument('--batch-size', '-btch', type=int, default=16, help='Select batch size.')
    args_parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='Select learning rate.')
    args_parser.add_argument('--optimizer', '-opt', type=str, default='adam',
                             choices=['adam', 'ftrl', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam'],
                             help='Select optimizer.')
    args_parser.add_argument('--activation', '-act', type=str, default='swish', choices=['relu', 'swish'],
                             help='Select leaky relu gradient.')
    args_parser.add_argument('--dropout', '-dor', type=float, default=0.5, help='Select dropout ratio.')
    args_parser.add_argument('--epochs', '-e', type=int, default=500, help='Number of max training epochs.')
    args_parser.add_argument('--early-stop', '-es', type=int, default=30, help='Number of early stopping epochs.')
    args_parser.add_argument('--test', '-test', action='store_true', help='Test loaded model with isic2020.')
    args_parser.add_argument('--load-model', '-load', type=str, help='Path to load model.')
    args_parser.add_argument('--fine', '-fine', action='store_true', help='Fine tune.')
    args_parser.add_argument('--strategy', '-strg', type=str, default='mirrored', choices=['mirrored', 'singlegpu'],
                             help='Select parallelization strategy.')
    args_parser.add_argument('--gpus', '-gpus', type=int, default=2, help='Select number of GPUs.')
    args_parser.add_argument('--os', '-os', type=str, default=sys.platform, help='Operating System.')
    return args_parser


class Directories:
    def __init__(self, args):
        self.trial_id = args['trial_id']
        self.task = args['task']
        self.image_type = args['image_type']
        self.load_model = args['load_model']
        self.fine = args['fine']
        self.test = args['test']
        self.image_size = args['image_size']
        self.new_folder = os.path.join(self.task, self.image_type, self.trial_id)
        self.proc_img_folder = os.path.join(MAIN_DIR, f"proc_{self.image_size}")
        self.dirs = self._dir_dict()

    def _dir_dict(self):
        """ Set paths for each run with unique path for example:
              logs/5cls/both/130921080509
            trials/5cls/both/130921080509
            models/5cls/both/130921080509/
            MAIN_PATH --- LOGS_DIR: Tensorboard logs
                      --- TRIALS_DIR: Hparams and results
                      --- MODELS_DIR: saved model
                  """

        directories: dict = {'init_img_folder': INIT_DATA_DIR,
                             'train': data_csv['train'],
                             'validation': data_csv['validation'],
                             'test': data_csv['test'],
                             'isic20_test': data_csv['isic20_test'],
                             'logs': os.path.join(LOGS_DIR, self.new_folder),
                             'trial': os.path.join(TRIALS_DIR, self.new_folder),
                             'save_path': os.path.join(MODELS_DIR, self.new_folder),
                             'load_path': os.path.join(MODELS_DIR, self.new_folder)}
        if os.getenv('SLURMD_NODENAME'):  # Append node name if training on HPC with SLURM.
            for fold in ('logs', 'trial', 'save_path'):
                directories[fold] = '-'.join([directories[fold], os.getenv('SLURMD_NODENAME')])
        if self.load_model:
            directories['load_path'] = self.load_model
        if self.fine:
            directories['logs'] = directories['logs'] + '_fine'
            directories['trial'] = directories['trial'] + '_fine'
            directories['save_path'] = directories['save_path'] + '_fine'

        directories['model_summary'] = os.path.join(directories['trial'], 'model_summary.txt')
        directories['train_logs'] = os.path.join(directories['trial'], 'train_logs.csv')
        directories['proc_img_folder'] = self.proc_img_folder
        directories['hparams_log'] = HPARAMS_FILE
        directories['data_info'] = INFO_DIR
        if not self.test:
            os.makedirs(directories['logs'], exist_ok=True)
            os.makedirs(directories['trial'], exist_ok=True)
        return directories


def log_params(args, dirs):
    if os.path.exists(path=dirs['hparams_log']):
        aw = 'a'
    else:
        aw = 'w'
    with open(dirs['hparams_log'], aw) as f:
        fieldnames = args.keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='', extrasaction='ignore')
        if aw == 'w':
            writer.writeheader()
        writer.writerows([args])
