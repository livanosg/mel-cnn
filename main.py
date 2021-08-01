import os
import argparse
from grid_search import grid


def parse_module():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', choices=['incept', 'xept', 'effnet0', 'effnet1'], default='effnet0', help='Select pretrained model.')
    parser.add_argument('--optimizer', '-opt', required=True, choices=['adam', 'ftrl', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam'], type=str, help='Select optimizer.')
    parser.add_argument('--image_size', '-is', required=True, type=int, help='Select image size.')
    parser.add_argument('--image_type', '-it', required=True, type=str, choices=['derm', 'clinic', 'both'], help='Select image type to use during training.')
    parser.add_argument('--colour', '-clr', default='rgb', type=str, help='Select image size.')
    parser.add_argument('--batch-size', '-btch', default=4, type=int, help='Select batch size.')
    parser.add_argument('--learning-rate', '-lr', default=1e-5, type=float, help='Select learning rate.')
    parser.add_argument('--dropout-ratio', '-dor', required=True, type=float, help='Select dropout ratio.')
    parser.add_argument('--relu-grad', '-rg', required=True, type=float, help='Select leaky relu gradient.')
    parser.add_argument('--dataset-frac', '-frac', default=1., type=float, help='Dataset fraction.')
    parser.add_argument('--epochs', '-e', required=True, type=int, help='Select epochs.')
    parser.add_argument('--early-stop', '-es', required=True, type=int, help='Select early stop epochs.')
    parser.add_argument('--nodes', '-nod', required=True, type=str, choices=['multi', 'one'], help='Select training nodes.')
    parser.add_argument('--mode', '-mod', required=True, type=str, choices=['5cls', 'ben_mal', 'nev_mel'], help='Select the type of outputs.')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Set verbosity.')
    parser.add_argument('--layers', '-lrs', default=1, type=int, help='Select set of layers.')
    return parser


if __name__ == '__main__':
    args = parse_module().parse_args().__dict__
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    # Set verbose for TF CPP LOG
    # 0 = all logs, 1 = filter out INFO, 2 = 1 + WARNING, 3 = 2 + ERROR
    os.environ['AUTOGRAPH_VERBOSITY'] = '1'
    if args['verbose'] > 3:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = f"{3 - args['verbose']}"
    # Set verbose for keras
    if args['verbose'] == 1:
        args['verbose'] = 2
    elif args['verbose'] >= 2:
        args['verbose'] = 1
    else:
        args['verbose'] = 0
    grid(args=args)
