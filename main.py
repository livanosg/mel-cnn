import os
from settings import parser, Directories, log_params
from preproc_images import setup_images
from train_script import train_fn, test_fn

if __name__ == '__main__':
    args = vars(parser().parse_args())
    dirs = Directories(args).dirs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, (range(args['gpus']))))
    # os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    # os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
    # os.environ['TF_XLA_FLAGS'] = f'--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices --tf_xla_cpu_global_jit'
    setup_images(args, dirs)
    if not args['test']:
        log_params(args, dirs)
        train_fn(args, dirs)
        args['load_model'] = args['trial_id']
        dirs = Directories(args).dirs
    if args['image_type'] != 'both':
        test_fn(args, dirs)
    else:
        for image_type in ('clinic', 'derm'):
            args['image_type'] = image_type
            test_fn(args, dirs)
exit()
