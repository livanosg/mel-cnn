import sklearn
import os
import argparse
from absl import logging
from data_check import check_create_dataset
from train_script import training
from config import directories, CLASS_NAMES


def parser():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--pretrained', '-pt', default='effnet1', choices=['incept', 'xept', 'effnet0', 'effnet1'], help='Select pretrained model.')
    args_parser.add_argument('--task', '-task', required=True, type=str, choices=['5cls', 'ben_mal', 'nev_mel'], help='Select the type of model.')
    args_parser.add_argument('--image-type', '-it', required=True, type=str, choices=['derm', 'clinic', 'both'], help='Select image type to use during training.')
    args_parser.add_argument('--image-size', '-is', default=500, type=int, help='Select image size.')
    args_parser.add_argument('--only-image', '-io', action='store_true', help='Train model only with images.')
    args_parser.add_argument('--colour', '-clr', default='rgb', type=str, help='Select image size.')
    args_parser.add_argument('--batch-size', '-btch', default=8, type=int, help='Select batch size.')
    args_parser.add_argument('--learning-rate', '-lr', default=1e-5, type=float, help='Select learning rate.')
    args_parser.add_argument('--optimizer', '-opt', default='adam', choices=['adam', 'ftrl', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam'], type=str, help='Select optimizer.')
    args_parser.add_argument('--activation', '-act', default='swish', choices=['relu', 'swish'], type=str, help='Select leaky relu gradient.')
    args_parser.add_argument('--dropout', '-dor', default=0.3, type=float, help='Select dropout ratio.')
    args_parser.add_argument('--epochs', '-e', default=500, type=int, help='Number of epochs epochs.')
    args_parser.add_argument('--layers', '-lrs', default=1, type=int, help='Select multiplier for inception layers\' nodes.')
    args_parser.add_argument('--no-image-weights', '-niw', action='store_true', help='Set to not weight per image type.')
    args_parser.add_argument('--no-image-type', '-nit', action='store_true', help='Set to remove image type from training.')
    args_parser.add_argument('--dataset-frac', '-frac', default=1., type=float, help='Dataset fraction.')
    args_parser.add_argument('--strategy', '-strg', default='mirrored', type=str, choices=['multiworker', 'mirrored'], help='Select parallelization strategy.')
    args_parser.add_argument('--validate', '-val', action='store_true', help='Validate model')
    args_parser.add_argument('--test', '-test', action='store_true', help='Test to isic2020.')
    args_parser.add_argument('--test-model', '-testmodel', type=str, help='Test to isic2020.')
    args_parser.add_argument('--verbose', '-v', default=0, action='count', help='Set verbosity.')
    return args_parser


if __name__ == '__main__':
    args = parser().parse_args().__dict__
    os.environ['AUTOGRAPH_VERBOSITY'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(max(0, (3 - args['verbose'])))  # 0 log all, 1:noINFO, 2:noWARNING, 3:noERROR
    if args['verbose'] >= 2:  # Set verbosity for keras 0 = silent, 1 = progress bar, 2 = one line per epoch.
        args['verbose'] = 1
    else:
        logging.set_verbosity(logging.INFO)  # Suppress > values
        args['verbose'] = 2

    args['dir_dict'] = directories(args=args)
    args['class_names'] = CLASS_NAMES[args['task']]
    args['num_classes'] = len(args['class_names'])
    args['input_shape'] = (args['image_size'], args['image_size'], 3)
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['OMP_NUM_THREADS'] = '1'
    for key, path in args['dir_dict']['data_csv'].items():
        check_create_dataset(key=key, datasplit_path=path, args=args)
    training(args=args)
    exit()
