import os
from contextlib import redirect_stdout

import tensorflow as tf

tf.get_logger().setLevel('INFO')

import tensorflow_addons as tfa
from custom_losses import categorical_focal_loss, losses
from custom_metrics import GeometricMean, calc_metrics
from data_prep import get_train_dataset, get_val_test_dataset, get_isic20_test_dataset
from features_def import TASK_CLASSES
from models_init import model_struct
from settings import parser, Directories, log_params

# from prepare_images import setup_images

args = vars(parser().parse_args())
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, (range(args['gpus']))))
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
if args['os'] == 'linux':
    # XLA currently ignores TF seeds to random operations. Workaround: use the recommended RNGs such as
    # tf.random.stateless_uniform or the tf.random.Generator directly.
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda/'
    os.environ['TF_XLA_FLAGS'] = f'--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices --tf_xla_cpu_global_jit'
    # From https://github.com/tensorflow/tensorflow/issues/44176#issuecomment-783768033
    # LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 python main.py args
dirs = Directories(args).dirs
# setup_images(args, dirs)
if args['os'] == 'win32':
    cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
else:
    cross_device_ops = tf.distribute.NcclAllReduce()
if args['strategy'] == 'mirrored':
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
else:
    strategy = tf.distribute.OneDeviceStrategy('GPU')
assert args['gpus'] == strategy.num_replicas_in_sync

if not args['test']:
    log_params(args, dirs)
    optimizer = {'adam': tf.keras.optimizers.Adam, 'adamax': tf.keras.optimizers.Adamax,
                 'nadam': tf.keras.optimizers.Nadam, 'ftrl': tf.keras.optimizers.Ftrl,
                 'rmsprop': tf.keras.optimizers.RMSprop, 'sgd': tf.keras.optimizers.SGD,
                 'adagrad': tf.keras.optimizers.Adagrad, 'adadelta': tf.keras.optimizers.Adadelta
                 }[args['optimizer']]
    loss = losses(args)[args['loss_fn']]
    with strategy.scope():
        if args['load_model']:
            model = tf.keras.models.load_model(dirs['load_path'], compile=True,
                                               custom_objects={'categorical_focal_loss_fixed': categorical_focal_loss(),
                                                               'GeometricMean': GeometricMean})
        else:
            model = model_struct(args=args)
        if args['fine']:
            for layer in model.layers:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
        else:
            for layer in model.layers:
                if layer.name.startswith(('efficient', 'inception', 'xception')):
                    layer.trainable = False

        model.compile(loss=loss, optimizer=optimizer(learning_rate=args['learning_rate'] * args['gpus']),
                      metrics=[tfa.metrics.F1Score(num_classes=len(TASK_CLASSES[args['task']]), average='macro', name='f1'),
                               GeometricMean()])

        with redirect_stdout(open(dirs['model_summary'], 'w', encoding='utf-8')):
            model.summary()  # show_trainable=True)

        model.fit(x=get_train_dataset(args=args, dirs=dirs), epochs=args['epochs'], verbose=2,
                  validation_data=get_val_test_dataset(args=args, dataset='validation', dirs=dirs),
                  callbacks=[tf.keras.callbacks.CSVLogger(filename=dirs['train_logs'], separator=',', append=True),
                             tf.keras.callbacks.EarlyStopping(monitor='val_geometric_mean', mode='max', verbose=1,
                                                              patience=args['early_stop'], restore_best_weights=True),
                             # EnrTensorboard(val_data=validation_data, log_dir=dirs['logs'], class_names=TASK_CLASSES[args['task']]),
                             ]
                  )
    model.save(filepath=dirs['save_path'])

args['clinic_val'] = False
for image_type in ('clinic', 'derm'):
    args['image_type'] = image_type
    thr_d, thr_f1 = calc_metrics(args=args, dirs=dirs, model=model,
                                 dataset=get_val_test_dataset(args=args, dataset='validation', dirs=dirs),
                                 dataset_name='validation')
    test_datasets = {'derm': ['isic16_test', 'isic17_test', 'isic18_val_test',
                              'mclass_derm_test', 'up_test'],
                     'clinic': ['up_test', 'dermofit_test', 'mclass_clinic_test']}
    if args['task'] == 'nev_mel':
        test_datasets['derm'].remove('isic16_test')

    for test_dataset in test_datasets[args['image_type']]:
        calc_metrics(args=args, dirs=dirs, model=model,
                     dataset=get_val_test_dataset(args=args, dataset=test_dataset, dirs=dirs),
                     dataset_name=test_dataset, dist_thresh=thr_d, f1_thresh=thr_f1)
    if args['task'] == 'ben_mal' and args['image_type'] == 'derm':
        calc_metrics(args=args, dirs=dirs, model=model,
                     dataset=get_isic20_test_dataset(args=args, dirs=dirs),
                     dataset_name='isic20_test', dist_thresh=thr_d, f1_thresh=thr_f1)

exit()
