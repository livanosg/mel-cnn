import os
import sys

from train_handle.custom_metrics import calc_metrics
from data_handle.data_prep import MelData
from settings import log_params
# from prepare_images import setup_images
from model_handle.setup_model import setup_model
from train_handle.train import train_fn

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, (range(args['gpus']))))
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    if sys.platform == 'linux':
        # XLA currently ignores TF seeds to random operations. Workaround: use the recommended RNGs such as
        # tf.random.stateless_uniform or the tf.random.Generator directly.
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
        os.environ['TF_XLA_FLAGS'] = f'--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices --tf_xla_cpu_global_jit'
        # From https://github.com/tensorflow/tensorflow/issues/44176#issuecomment-783768033
        # LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 python main.py args
    trial_id = ''
    task = ''
    image_type = ''
    load_model = ''
    fine = ''
    test = ''
    image_size = ''
    mode = ''
    image_type = ''
    clinic_val = ''
    # setup_images(args, dirs)
    data = MelData(task, mode, image_type)
    model, strategy = setup_model(*args)
    if not args['test']:
        log_params(args, dirs)
        train_fn(args, dirs, data, model, strategy)
    args['clinic_val'] = False
    for image_type in ('clinic', 'derm'):  # Test
        args['image_type'] = image_type
        thr_d, thr_f1 = calc_metrics(args=args, dirs=dirs, model=model,
                                     dataset=data.get_dataset(dataset='validation'),
                                     dataset_name='validation')
        test_datasets = {'derm': ['isic16_test', 'isic17_test', 'isic18_val_test',
                                  'mclass_derm_test', 'up_test'],
                         'clinic': ['up_test', 'dermofit_test', 'mclass_clinic_test']}
        if args['task'] == 'nev_mel':
            test_datasets['derm'].remove('isic16_test')

        for test_dataset in test_datasets[args['image_type']]:
            calc_metrics(args=args, dirs=dirs, model=model,
                         dataset=data.get_dataset(dataset=test_dataset),
                         dataset_name=test_dataset, dist_thresh=thr_d, f1_thresh=thr_f1)
        if args['task'] == 'ben_mal':
            calc_metrics(args=args, dirs=dirs, model=model,
                         dataset=data.get_dataset(dataset='isic20_test'),
                         dataset_name='isic20_test', dist_thresh=thr_d, f1_thresh=thr_f1)
    exit()
