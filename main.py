import gc
import os

from data_prep import MelData
from settings import parser, Directories, log_params
from prepare_images import setup_images
from setup_model import setup_model
from train import train_fn
from test import test_fn

if __name__ == '__main__':
    args = vars(parser().parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, (range(args['gpus']))))
    # os.environ["OMP_NUM_THREADS"] = '1'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    if args['os'] == 'linux':
        pass
        # XLA currently ignores TF seeds to random operations. Workaround: use the recommended RNGs such as
        # tf.random.stateless_uniform or the tf.random.Generator directly.
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
        os.environ['TF_XLA_FLAGS'] = f'--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices --tf_xla_cpu_global_jit'

    dirs = Directories(args).dirs
    setup_images(args, dirs)
    data = MelData(args, dirs)
    model, strategy = setup_model(args, dirs)
    if not args['test']:
        log_params(args, dirs)
        model = train_fn(args, dirs, data, model, strategy)
    args['clinic_val'] = False
    for image_type in ('clinic', 'derm'):
        args['image_type'] = image_type
        data = MelData(args, dirs)
        test_fn(args, dirs, data, model)
    del dirs
    del data
    del model
    del strategy
    del args
    gc.collect()
exit()
