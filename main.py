import sklearn
import os
import argparse
from absl import logging
from data_check import check_create_dataset
from train_script import train_val_test
from config import dir_dict, TASK_CLASSES


def parser():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--pretrained', '-pt', type=str, default='effnet1', choices=['incept', 'xept', 'effnet0', 'effnet1', 'effnet6'], help='Select pretrained model.')
    args_parser.add_argument('--task', '-task', type=str, required=True, choices=['5cls', 'ben_mal', 'nev_mel'], help='Select the type of model.')
    args_parser.add_argument('--image-type', '-it', type=str, required=True, choices=['derm', 'clinic', 'both'], help='Select image type to use during training.')
    args_parser.add_argument('--image-size', '-is', type=int, default=500, help='Select image size.')
    args_parser.add_argument('--only-image', '-io', action='store_true', help='Train model only with images.')
    args_parser.add_argument('--colour', '-clr', type=str, default='rgb', help='Select image size.')
    args_parser.add_argument('--batch-size', '-btch', type=int, default=16, help='Select batch size.')
    args_parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4, help='Select learning rate.')
    args_parser.add_argument('--optimizer', '-opt', type=str, default='adamax', choices=['adam', 'ftrl', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam'], help='Select optimizer.')
    args_parser.add_argument('--activation', '-act', type=str, default='swish', choices=['relu', 'swish'], help='Select leaky relu gradient.')
    args_parser.add_argument('--dropout', '-dor', type=float, default=0.2, help='Select dropout ratio.')
    args_parser.add_argument('--epochs', '-e', type=int, default=500, help='Number of epochs epochs.')
    args_parser.add_argument('--loss-fn', '-loss', type=str, default='cxe', choices=['cxe', 'focal', 'perclass'], help='Select loss function.')
    args_parser.add_argument('--conv_layers', '-clrs', type=int, default=32, help='Select multiplier for number of nodes in inception layers.')
    args_parser.add_argument('--dense-layers', '-dlrs', type=int, default=16, help='Select multiplier for number of nodes in dense layers.')
    args_parser.add_argument('--merge-layers', '-mlrs', type=int, default=8, help='Select multiplier for number of nodes in merge layers.')
    args_parser.add_argument('--no-image-weights', '-niw', action='store_true', help='Set to not weight per image type.')
    args_parser.add_argument('--no-image-type', '-nit', action='store_true', help='Set to remove image type from training.')
    args_parser.add_argument('--dataset-frac', '-frac', type=float, default=1., help='Dataset fraction.')
    args_parser.add_argument('--strategy', '-strg', type=str, default='mirrored', choices=['multiworker', 'mirrored', 'singlegpu'], help='Select parallelization strategy.')
    args_parser.add_argument('--validate', '-val', action='store_true', help='Validate model')
    args_parser.add_argument('--test', '-test', action='store_true', help='Test to isic2020.')
    args_parser.add_argument('--test-model', '-testmodel', type=str, help='Path to load model.')
    args_parser.add_argument('--verbose', '-v', default=0, action='count', help='Set verbosity.')
    return args_parser


if __name__ == '__main__':
    args = parser().parse_args().__dict__
    os.environ['AUTOGRAPH_VERBOSITY'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(max(0, (3 - args['verbose'])))  # 0 log all, 1:noINFO, 2:noWARNING, 3:noERROR
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    if args['verbose'] >= 2:  # Set verbosity for keras 0 = silent, 1 = progress bar, 2 = one line per epoch.
        args['verbose'] = 1
    else:
        logging.set_verbosity(logging.INFO)  # Suppress > values
        args['verbose'] = 2

    args['dir_dict'] = dir_dict(args=args)
    args['class_names'] = TASK_CLASSES[args['task']]
    args['num_classes'] = len(args['class_names'])
    args['input_shape'] = (args['image_size'], args['image_size'], 3)
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['OMP_NUM_THREADS'] = '1'
    for key, path in args['dir_dict']['data_csv'].items():
        check_create_dataset(key=key, datasplit_path=path, args=args)
    train_val_test(args=args)
    exit()
