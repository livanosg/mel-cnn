import os
import argparse
from tensorflow.keras.applications import xception, inception_v3, efficientnet
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
    args_parser.add_argument('--activation', '-act', default='relu', choices=['relu', 'swish'], type=str, help='Select leaky relu gradient.')
    args_parser.add_argument('--dataset-frac', '-frac', default=1., type=float, help='Dataset fraction.')
    args_parser.add_argument('--epochs', '-e', default=500, type=int, help='Select epochs.')
    args_parser.add_argument('--early-stop', '-es', default=30, type=int, help='Select early stop epochs.')
    args_parser.add_argument('--strategy', '-strg', default='mirrored', type=str, choices=['multiworker', 'mirrored'], help='Select training nodes.')
    args_parser.add_argument('--mode', '-mod', required=True, type=str, choices=['5cls', 'ben_mal', 'nev_mel'], help='Select the type of outputs.')
    args_parser.add_argument('--verbose', '-v', default=0, action='count', help='Set verbosity.')
    args_parser.add_argument('--layers', '-lrs', default=2, type=int, help='Select set of layers.')
    args_parser.add_argument('--validate', '-val', action='store_true', help='Test to isic2020.')
    args_parser.add_argument('--test', '-test', action='store_true', help='Test to isic2020.')
    args_parser.add_argument('--test-model', '-tstmdl', type=str, help='Test to isic2020.')
    return args_parser


if __name__ == '__main__':
    args = parser().parse_args().__dict__
    args['dir_dict'] = directories(args=args)
    check_create_dataset(args=args, force=True)
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

    if args["strategy"] == 'multiworker':
        for i in ['http_proxy', 'https_proxy', 'http', 'https']:
            try:
                del os.environ[i]
                print(f'{i} unset')
            except KeyError:
                pass
        slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
        args["dir_dict"]["save_path"] += f"-{slurm_resolver.task_id}-{slurm_resolver.task_type}"
        strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()
    args['class_names'] = CLASS_NAMES[args["mode"]]
    args['num_classes'] = len(args['class_names'])
    args['input_shape'] = (args['image_size'], args['image_size'], 3)
    args['batch_size'] = args['batch_size'] * strategy.num_replicas_in_sync  # Global Batch
    args["learning_rate"] = args["learning_rate"] * strategy.num_replicas_in_sync
    models = {'xept': xception.Xception, 'incept': inception_v3.InceptionV3,
              'effnet0': efficientnet.EfficientNetB0, 'effnet1': efficientnet.EfficientNetB1}
    preproc_input_fn = {'xept':  xception.preprocess_input, 'incept': inception_v3.preprocess_input,
                        'effnet0': efficientnet.preprocess_input, 'effnet1':  efficientnet.preprocess_input}
    args['preprocess_fn'] = preproc_input_fn[args['model']]
    args['model'] = models[args['model']]
    datasets = MelData(args=args)
    args['train_data'] = datasets.get_dataset(mode='train')
    args['val_data'] = datasets.get_dataset(mode='val')
    args['test_data'] = datasets.get_dataset(mode='test')
    args['isic20_test'] = datasets.get_dataset(mode='isic20_test')
    if not (args['test'] or args['validate']):
        with open(args['dir_dict']['hparams_logs'], 'a') as f:
            [f.write(f"{key.capitalize()}: {args[key]}\n") for key in args.keys() if key not in ('dir_dict', 'hparams')]
            f.write('Directories\n')
            [f.write(f"{key.capitalize()}: {args['dir_dict'][key]}\n") for key in args['dir_dict'].keys()]
            f.write(f"Number of replicas in sync: {strategy.num_replicas_in_sync}\n")
            f.write(datasets.info())
    if args['test'] or args['validate']:
        args['dir_dict']["save_path"] = args['test_model']
        args['dir_dict']['trial'] = os.path.dirname(os.path.dirname(args['dir_dict']["save_path"]))

        model = tf.keras.models.load_model(args["dir_dict"]["save_path"])
        if args['test']:
            calc_metrics(args=args, model=model, dataset=args['isic20_test'], dataset_type='isic20_test')
        elif args['validate']:
            args['dir_dict']["save_path"] = args['test_model']
            calc_metrics(args=args, model=model, dataset=args['val_data'], dataset_type='validation')
            calc_metrics(args=args, model=model, dataset=args['test_data'], dataset_type='test')
    else:
        with strategy.scope():
            training(args=args)
    exit()

