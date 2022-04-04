import os
import argparse
from datetime import datetime

from absl import logging as absl_log
from data_pipe import MelData
from preproc_images import setup_images
from train_script import train_fn, test_fn
from config import dir_dict, TASK_CLASSES, log_params


def parser():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--pretrained', '-pt', type=str, default='effnet1', choices=['incept', 'xept', 'effnet0', 'effnet1', 'effnet6'], help='Select pretrained model.')
    args_parser.add_argument('--task', '-task', type=str, required=True, choices=['5cls', 'ben_mal', 'nev_mel'], help='Select the type of model.')
    args_parser.add_argument('--image-type', '-it', type=str, required=True, choices=['derm', 'clinic', 'both'], help='Select image type to use during training.')
    args_parser.add_argument('--image-size', '-is', type=int, default=224, help='Select image size.')
    args_parser.add_argument('--no-clinical-data', '-ncd', action='store_true', help='Train model only with images.')
    args_parser.add_argument('--no-image-type', '-nit', action='store_true', help='Set to remove image type from training.')
    args_parser.add_argument('--batch-size', '-btch', type=int, default=16, help='Select batch size.')
    args_parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='Select learning rate.')
    args_parser.add_argument('--optimizer', '-opt', type=str, default='adam', choices=['adam', 'ftrl', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam'], help='Select optimizer.')
    args_parser.add_argument('--activation', '-act', type=str, default='swish', choices=['relu', 'swish'], help='Select leaky relu gradient.')
    args_parser.add_argument('--dropout', '-dor', type=float, default=0.5, help='Select dropout ratio.')
    args_parser.add_argument('--epochs', '-e', type=int, default=500, help='Number of epochs epochs.')
    args_parser.add_argument('--loss-fn', '-loss', type=str, default='cxe', choices=['cxe', 'focal', 'perclass','wcxe', 'combined'], help='Select loss function.')
    args_parser.add_argument('--loss-frac', '-lossf', type=float, default=.5, help='log_dice_loss ratio in custom loss.')
    args_parser.add_argument('--weighted-loss', '-wl', action='store_true', help='Apply class weights.')
    args_parser.add_argument('--weighted-samples', '-ws', action='store_true', help='Apply sample weights per image type.')
    args_parser.add_argument('--conv_layers', '-clrs', type=int, default=32, help='Select multiplier for number of nodes in inception layers.')
    args_parser.add_argument('--dense-layers', '-dlrs', type=int, default=16, help='Select multiplier for number of nodes in dense layers.')
    args_parser.add_argument('--merge-layers', '-mlrs', type=int, default=32, help='Select multiplier for number of nodes in merge layers.')
    args_parser.add_argument('--dataset-frac', '-frac', type=float, default=1., help='Dataset fraction.')
    args_parser.add_argument('--strategy', '-strg', type=str, default='singlegpu', choices=['mirrored', 'singlegpu'], help='Select parallelization strategy.')
    args_parser.add_argument('--load-model', '-load', type=str, help='Path to load model.')
    args_parser.add_argument('--test', '-test', action='store_true', help='Test loaded model with isic2020.')
    args_parser.add_argument('--fine', '-fine', action='store_true', help='Fine tune.')
    args_parser.add_argument('--gpus', '-gpus', type=int, default=2, help='Select number of GPUs.')
    return args_parser


if __name__ == '__main__':
    args = parser().parse_args().__dict__
    os.environ['AUTOGRAPH_VERBOSITY'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 log all, 1:noINFO, 2:noWARNING, 3:noERROR
    absl_log.set_verbosity(absl_log.INFO)
    args['trial_id'] = datetime.now().strftime('%d%m%y%H%M%S')
    args['dir_dict'] = dir_dict(args=args)
    args['class_names'] = TASK_CLASSES[args['task']]
    args['num_classes'] = len(args['class_names'])
    args['input_shape'] = (args['image_size'], args['image_size'], 3)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, (range(args['gpus']))))
    # os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    # os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/apps/compilers/cuda/10.1.168'
    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices --tf_xla_cpu_global_jit'
    # os.environ['OMP_NUM_THREADS'] = '1'
    print('Setting up Datasets...')
    for key, path in args['dir_dict']['data_csv'].items():
        setup_images(csv_path=path, args=args)
    print('Done!')

    if not args['test']:
        log_params(args=args)
        train_fn(args=args)
        if not args['load_model']:
            args['load_model'] = args['dir_dict']['save_path']
    if args['image_type'] != 'both':
        test_fn(args=args)
    else:
        for image_type in ('clinic', 'derm'):
            args['image_type'] = image_type
            test_fn(args=args)


exit()
