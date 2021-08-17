import os
import argparse
from config import directories, CLASS_NAMES
from data_pipe import MelData
from data_prep import check_create_dataset
from metrics import calc_metrics
from train_script import training
import tensorflow as tf


def parser():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--model', '-m', default='effnet1', choices=['incept', 'xept', 'effnet0', 'effnet1'], help='Select pretrained model.')
    args_parser.add_argument('--optimizer', '-opt', default='adamax', choices=['adam', 'ftrl', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam'], type=str, help='Select optimizer.')
    args_parser.add_argument('--image_size', '-is', default=500, type=int, help='Select image size.')
    args_parser.add_argument('--image_type', '-it', required=True, type=str, choices=['derm', 'clinic', 'both'], help='Select image type to use during training.')
    args_parser.add_argument('--only-image', '-io', action='store_true', help='Test to isic2020.')
    args_parser.add_argument('--colour', '-clr', default='rgb', type=str, help='Select image size.')
    args_parser.add_argument('--batch-size', '-btch', default=8, type=int, help='Select batch size.')
    args_parser.add_argument('--learning-rate', '-lr', default=1e-5, type=float, help='Select learning rate.')
    args_parser.add_argument('--dropout', '-dor', default=0.2, type=float, help='Select dropout ratio.')
    args_parser.add_argument('--activation', '-act', default='swish', choices=['relu', 'swish'], type=str, help='Select leaky relu gradient.')
    args_parser.add_argument('--dataset-frac', '-frac', default=1., type=float, help='Dataset fraction.')
    args_parser.add_argument('--epochs', '-e', default=500, type=int, help='Select epochs.')
    args_parser.add_argument('--early-stop', '-es', default=30, type=int, help='Select early stop epochs.')
    args_parser.add_argument('--strategy', '-strg', default='mirrored', type=str, choices=['multiworker', 'mirrored'], help='Select training nodes.')
    args_parser.add_argument('--mode', '-mod', required=True, type=str, choices=['5cls', 'ben_mal', 'nev_mel'], help='Select the type of outputs.')
    args_parser.add_argument('--verbose', '-v', default=0, action='count', help='Set verbosity.')
    args_parser.add_argument('--layers', '-lrs', default=2, type=int, help='Select set of layers.')
    args_parser.add_argument('--validate', '-val', action='store_true', help='Test to isic2020.')
    args_parser.add_argument('--test', '-test', action='store_true', help='Test to isic2020.')
    args_parser.add_argument('--test-model', '-testmodel', type=str, help='Test to isic2020.')
    return args_parser


if __name__ == '__main__':
    args = parser().parse_args().__dict__
    args['dir_dict'] = directories(args=args)
    args['class_names'] = CLASS_NAMES[args["mode"]]
    args['num_classes'] = len(args['class_names'])
    args['input_shape'] = (args['image_size'], args['image_size'], 3)

    check_create_dataset(args=args, force=True)
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['AUTOGRAPH_VERBOSITY'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = f"{max(0,(3 - args['verbose']))}"  # 0 log all, 1:noINFO, 2:noWARNING, 3:noERROR
    if args['verbose'] >= 2:  # Set verbosity for keras 0 = silent, 1 = progress bar, 2 = one line per epoch.
        args['verbose'] = 1
    else:
        args['verbose'] = 2

    if args["strategy"] == 'multiworker':
        slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
        args["dir_dict"]["save_path"] += f"-{slurm_resolver.task_id}-{slurm_resolver.task_type}"
        strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()
    args['replicas'] = strategy.num_replicas_in_sync
    datasets = MelData(args=args)
    args['train_data'] = datasets.get_dataset(mode='train')
    args['val_data'] = datasets.get_dataset(mode='val')
    args['test_data'] = datasets.get_dataset(mode='test')
    args['isic20_test'] = datasets.get_dataset(mode='isic20_test')
    if args['test'] or args['validate']:
        args['dir_dict']["save_path"] = args['test_model']
        args['dir_dict']['trial'] = os.path.dirname(os.path.dirname(args['dir_dict']["save_path"]))
        model = tf.keras.models.load_model(args["dir_dict"]["save_path"])
        if args['test']:
            calc_metrics(args=args, model=model, dataset=args['isic20_test'], dataset_type='isic20_test')
        else:
            args['dir_dict']["save_path"] = args['test_model']
            calc_metrics(args=args, model=model, dataset=args['val_data'], dataset_type='validation')
            calc_metrics(args=args, model=model, dataset=args['test_data'], dataset_type='test')
    else:
        with open(args['dir_dict']['hparams_logs'], 'a') as f:
            [f.write(f"{': '.join([key.capitalize().rjust(25), str(args[key])])}\n") for key in args.keys() if key not in ('dir_dict', 'hparams',
                                                                                                                      'train_data', 'val_data',
                                                                                                                      'test_data', 'isic20_test')]
            f.write(f"Number of replicas in sync: {strategy.num_replicas_in_sync}\n")
            f.write(datasets.info())
        training(args=args, strategy=strategy)
        model = tf.keras.models.load_model(args['dir_dict']['save_path'])
        calc_metrics(args=args, model=model, dataset=args['test_data'], dataset_type='test')
        calc_metrics(args=args, model=model, dataset=args['val_data'], dataset_type='val')
        if args['mode'] == 'ben_mal':
            calc_metrics(args=args, model=model, dataset=args['isic20_test'], dataset_type='isic20_test')
    exit()
