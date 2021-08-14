import os
import argparse

from tensorflow.keras.applications import xception, inception_v3, efficientnet
from config import directories, CLASS_NAMES
from data_prep import check_create_dataset
from test_isic20 import test_isic20
from train_script import training
import tensorflow as tf


def parse_module():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='effnet1', choices=['incept', 'xept', 'effnet0', 'effnet1'], help='Select pretrained model.')
    parser.add_argument('--optimizer', '-opt', default='adamax', choices=['adam', 'ftrl', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam'], type=str, help='Select optimizer.')
    parser.add_argument('--image_size', '-is', default=500, type=int, help='Select image size.')
    parser.add_argument('--image_type', '-it', required=True, type=str, choices=['derm', 'clinic', 'both'], help='Select image type to use during training.')
    parser.add_argument('--only-image', '-io', action='store_true', help='Test to isic2020.')
    parser.add_argument('--colour', '-clr', default='rgb', type=str, help='Select image size.')
    parser.add_argument('--batch-size', '-btch', default=8, type=int, help='Select batch size.')
    parser.add_argument('--learning-rate', '-lr', default=1e-5, type=float, help='Select learning rate.')
    parser.add_argument('--dropout', '-dor', default=0.2, type=float, help='Select dropout ratio.')
    parser.add_argument('--activation', '-act', default='relu', choices=['relu', 'swish'], type=str, help='Select leaky relu gradient.')
    parser.add_argument('--dataset-frac', '-frac', default=1., type=float, help='Dataset fraction.')
    parser.add_argument('--epochs', '-e', default=500, type=int, help='Select epochs.')
    parser.add_argument('--early-stop', '-es', default=30, type=int, help='Select early stop epochs.')
    parser.add_argument('--nodes', '-nod', default='one', type=str, choices=['multi', 'one'], help='Select training nodes.')
    parser.add_argument('--mode', '-mod', required=True, type=str, choices=['5cls', 'ben_mal', 'nev_mel'], help='Select the type of outputs.')
    parser.add_argument('--verbose', '-v', default=0, action='count', help='Set verbosity.')
    parser.add_argument('--layers', '-lrs', default=2, type=int, help='Select set of layers.')
    parser.add_argument('--test', '-test', action='store_true', help='Test to isic2020.')
    parser.add_argument('--test-model', '-tstmdl', type=str, help='Test to isic2020.')
    return parser


if __name__ == '__main__':
    args = parse_module().parse_args().__dict__
    args['dir_dict'] = directories(args=args)
    check_create_dataset(args=args, force=True)
    try:
        if int(os.environ['SLURM_STEP_TASKS_PER_NODE']) > 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = f"{os.environ['SLURM_PROCID']}"
    except KeyError:
        pass
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['AUTOGRAPH_VERBOSITY'] = '1'

    if args['verbose'] > 3:  # 0 = all logs, 1 = filter out INFO, 2 = 1 + WARNING, 3 = 2 + ERROR
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = f"{3 - args['verbose']}"
    # Set verbosity for keras
    if args['verbose'] == 1:
        args['verbose'] = 2
    elif args['verbose'] >= 2:
        args['verbose'] = 1
    else:
        args['verbose'] = 0

    assert args["nodes"] in ("multi", "one")
    if args["nodes"] == 'multi':
        slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
        args["dir_dict"]["save_path"] += f"-{slurm_resolver.task_id}-{slurm_resolver.task_type}"
        args['strategy'] = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
    else:
        args['strategy'] = tf.distribute.MirroredStrategy()

    if args['nodes'] == 'multi':
        for i in ['http_proxy', 'https_proxy', 'http', 'https']:
            try:
                del os.environ[i]
                print(f'{i} unset')
            except KeyError:
                pass

    with open(args['dir_dict']['hparams_logs'], 'a') as f:
        [f.write(f"{key.capitalize()}: {args[key]}\n") for key in args.keys() if key not in ('dir_dict', 'hparams')]
        f.write('Directories\n')
        [f.write(f"{key.capitalize()}: {args['dir_dict'][key]}\n") for key in args['dir_dict'].keys()]
    args['class_names'] = CLASS_NAMES[args["mode"]]
    args['num_classes'] = len(args['class_names'])
    args['input_shape'] = (args['image_size'], args['image_size'], 3)
    args['batch_size'] = args['batch_size'] * args['strategy'].num_replicas_in_sync  # Global Batch
    args["learning_rate"] = args["learning_rate"] * args['strategy'].num_replicas_in_sync
    models = {'xept': xception.Xception, 'incept': inception_v3.InceptionV3,
              'effnet0': efficientnet.EfficientNetB0, 'effnet1': efficientnet.EfficientNetB1}
    preproc_input_fn = {'xept':  xception.preprocess_input, 'incept': inception_v3.preprocess_input,
                        'effnet0': efficientnet.preprocess_input, 'effnet1':  efficientnet.preprocess_input}
    args['preprocess_fn'] = preproc_input_fn[args['model']]
    args['model'] = models[args['model']]

    if args['test']:
        print('Testing')
        args['dir_dict']["save_path"] = args['test_model']
        test_isic20(args=args)
    else:
        training(args=args)
    exit()
