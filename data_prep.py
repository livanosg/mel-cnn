import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import tensorflow_addons as tfa
from features_def import BEN_MAL_MAP, LOCATIONS, IMAGE_TYPE, SEX, AGE_APPROX, TASK_CLASSES
from settings import data_csv


def _prep_df(args, dataset: str, dirs):
    df = pd.read_csv(data_csv[dataset])
    # _log_info(args, dataset, df, dirs)

    if float(args['dataset_frac']) != 1.:
        if dataset in ('train', 'validation'):
            df = df.sample(frac=args['dataset_frac'])
    else:
        if dataset in 'train':
            df = df.sample(frac=args['dataset_frac'])

    # Set proper folder to fetch images
    df['image'] = df['image'].apply(lambda x: os.path.join(dirs['proc_img_folder'], x))
    # Handle NaNs
    for cat in ['location', 'sex', 'image_type']:
        df[cat] = df[cat].fillna('')
    df['age_approx'] = df['age_approx'].fillna(-1)
    df['age_approx'] = df['age_approx'].astype(int).astype('string')
    # df['age_approx'] = df['age_approx']
    # Define classes according to task
    if 'class' in df.columns:
        if args['task'] == 'ben_mal':
            df = df.replace(to_replace=BEN_MAL_MAP)
        if args['task'] == 'nev_mel':
            df = df.drop(df[~df['class'].isin(TASK_CLASSES[args['task']])].index, errors='ignore')
        if args['task'] == '5cls':  # Drop unclassified benign samples
            df = df.drop(df[df['class'].isin(['UNK'])].index, errors='ignore')

    if dataset == 'validation':
        if args['clinic_val']:  # Run validation on clinical dataset regardless the training image type
            df = df.drop(df[df['image_type'].isin(['derm'])].index, errors='ignore')
        elif args['image_type'] != 'both':  # Run validation on the same training image type
            df = df.drop(df[~df['image_type'].isin([args['image_type']])].index, errors='ignore')
    else:  # Keep dermoscopy or clinical image samples for the rest datasets according to training image type
        if args['image_type'] != 'both':
            df = df.drop(df[~df['image_type'].isin([args['image_type']])].index, errors='ignore')
    return df


def _prep_df_for_tfdataset(args, dataset, dirs):
    df = _prep_df(args, dataset, dirs)
    categories = [LOCATIONS, SEX, AGE_APPROX]
    columns = ['location', 'sex', 'age_approx']
    if not args['no_image_type']:
        categories.append(IMAGE_TYPE)
        columns.append('image_type')
    categories.append(TASK_CLASSES[args['task']])
    columns.append('class')
    features_env = OneHotEncoder(handle_unknown='ignore', categories=categories)
    features_env.fit(df[columns])
    ohe_data = features_env.transform(df[columns]).toarray()
    sample_weight = None
    if dataset == 'train':
        if args['image_type'] == 'both' and args['weighted_samples']:  # Sample weight for image type
            image_type_ohe = ohe_data[:, -4:-2]
            sample_weight = np.divide(np.amax(np.sum(image_type_ohe, axis=0)), np.sum(image_type_ohe, axis=0))
            sample_weight = np.sum(np.multiply(sample_weight, image_type_ohe), axis=-1)
        if args['weighted_loss']:  # Class weight
            class_weight = np.divide(np.amax(np.sum(ohe_data[:, -2:], axis=0)),
                                     np.sum(ohe_data[:, -2:], axis=0))
            class_weight = np.sum(np.multiply(class_weight, ohe_data[:, -2:]), axis=-1)

            if sample_weight is not None:  # From keras: `class_weight` and `sample_weight` are multiplicative.
                sample_weight = sample_weight * class_weight
            else:
                sample_weight = class_weight

        if sample_weight is None:  # Set sample weight to one if not set.
            sample_weight = np.ones(len(df), dtype=np.float32)
    if not args['no_clinical_data']:
        clinical_data = ohe_data[:, :-2]
    else:
        clinical_data = ohe_data
    #      image_path,         clinical_data, class,            sample weight
    return df['image'].values, clinical_data, ohe_data[:, -2:], sample_weight


def _read_images(image):
    return tf.cast(x=tf.io.decode_image(tf.io.read_file(tf.squeeze(image)), channels=3), dtype=tf.float32)


def get_train_dataset(args, dirs):
    rng = tf.random.Generator.from_non_deterministic_state()
    image_path, onehot_features, onehot_label, sample_weight = _prep_df_for_tfdataset(args, 'train', dirs)
    image_path_ds = tf.data.Dataset.from_tensor_slices(image_path)
    images_ds = image_path_ds.map(_read_images, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    image_path_ds = image_path_ds.batch(args['batch_size'] * args['gpus'], num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    images_ds = images_ds.batch(args['batch_size'] * args['gpus'], num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    images_ds = images_ds.map(lambda sample: augm(sample, args, rng), num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    onehot_features_ds = tf.data.Dataset.from_tensor_slices(onehot_features).batch(args['batch_size'] * args['gpus'], num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    onehot_label_ds = tf.data.Dataset.from_tensor_slices(onehot_label).batch(args['batch_size'] * args['gpus'], num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    sample_weight_ds = tf.data.Dataset.from_tensor_slices(sample_weight).batch(args['batch_size'] * args['gpus'], num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = tf.data.Dataset.zip((image_path_ds, images_ds, onehot_features_ds, onehot_label_ds, sample_weight_ds))
    ds = ds.map(lambda a, b, c, d, e: ({'image_path': a, 'image': b, 'clinical_data': c}, {'class': d}, e))
    # ds = ds.batch(args['batch_size'] * args['gpus'], deterministic=True)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    return ds.prefetch(tf.data.AUTOTUNE)


def get_val_test_dataset(args, dataset, dirs):
    image_path, onehot_features, onehot_label, sample_weight = _prep_df_for_tfdataset(args, dataset, dirs)
    image_path_ds = tf.data.Dataset.from_tensor_slices(image_path)
    images_ds = image_path_ds.map(_read_images, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    image_path_ds = tf.data.Dataset.from_tensor_slices(image_path).batch(50 * args['batch_size'] * args['gpus'], num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    images_ds = images_ds.batch(50 * args['batch_size'] * args['gpus'], num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    onehot_features_ds = tf.data.Dataset.from_tensor_slices(onehot_features).batch(50 * args['batch_size'] * args['gpus'], deterministic=True)
    onehot_label_ds = tf.data.Dataset.from_tensor_slices(onehot_label).batch(50 * args['batch_size'] * args['gpus'], deterministic=True)
    ds = tf.data.Dataset.zip((image_path_ds, images_ds, onehot_features_ds, onehot_label_ds))
    ds = ds.map(lambda a, b, c, d: ({'image_path': a, 'image': b, 'clinical_data': c}, {'class': d}))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    return ds.prefetch(tf.data.AUTOTUNE)


def get_isic20_test_dataset(args, dirs):
    image_path, onehot_features, onehot_label, sample_weight = _prep_df_for_tfdataset(args, 'isic20_test', dirs)
    image_path_ds = tf.data.Dataset.from_tensor_slices(image_path)
    images_ds = image_path_ds.map(_read_images, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    image_path_ds = tf.data.Dataset.from_tensor_slices(image_path).batch(50 * args['batch_size'] * args['gpus'], num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    images_ds = images_ds.batch(50 * args['batch_size'] * args['gpus'], num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    onehot_features_ds = tf.data.Dataset.from_tensor_slices(onehot_features).batch(50 * args['batch_size'] * args['gpus'], num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = tf.data.Dataset.zip((image_path_ds, images_ds, onehot_features_ds))
    ds = ds.map(lambda a, b, c: ({'image_path': a, 'image': b, 'clinical_data': c}))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    return ds.prefetch(tf.data.AUTOTUNE)


def augm(image, args, rng):
    image = tf.image.random_flip_up_down(tf.image.random_flip_left_right(image=image))
    image = tf.image.random_brightness(image=tf.image.random_contrast(image=image, lower=.5, upper=1.5), max_delta=60.)
    image = tf.image.random_saturation(image=tf.clip_by_value(image, clip_value_min=0., clip_value_max=255.), lower=0.8,
                                     upper=1.2)
    # _sharpness_image -> image_channels = tf.shape(image)[-1]
    image = tfa.image.sharpness(image=image, factor=rng.uniform(shape=[1], minval=0.5, maxval=1.5), name='Sharpness')
    image = tfa.image.translate(images=image, translations=rng.uniform(shape=[2],
                                                                   minval=-args['image_size'] * 0.2,
                                                                   maxval=args['image_size'] * 0.2,
                                                                   dtype=tf.float32), name='Translation')
    image = tfa.image.rotate(images=image, angles=tf.cast(rng.uniform(shape=[], dtype=tf.int32,
                                                                  minval=0, maxval=360), dtype=tf.float32),
                           interpolation='bilinear', name='Rotation')
    image = tf.cond(tf.less(rng.uniform(shape=[1]), 0.5),
                  lambda: tfa.image.gaussian_filter2d(image=image, sigma=1.5, filter_shape=3, name='Gaussian_filter'),
                  lambda: image)
    cutout_ratio = 0.15
    for i in range(3):
        mask_height = tf.cast(rng.uniform(shape=[], minval=0, maxval=args['image_size'] * cutout_ratio),
                              dtype=tf.int32) * 2
        mask_width = tf.cast(rng.uniform(shape=[], minval=0, maxval=args['image_size'] * cutout_ratio),
                             dtype=tf.int32) * 2
        image = tfa.image.random_cutout(image, mask_size=(mask_height, mask_width))
    image = {'xept': tf.keras.applications.xception.preprocess_input,
                       'incept': tf.keras.applications.inception_v3.preprocess_input,
                       'effnet0': tf.keras.applications.efficientnet.preprocess_input,
                       'effnet1': tf.keras.applications.efficientnet.preprocess_input,
                       'effnet6': tf.keras.applications.efficientnet.preprocess_input
             }[args['pretrained']](image)
    return image


def _log_info(args, dataset, df, dirs):
    class_names = TASK_CLASSES[args['task']]

    """log datasets general information"""
    if dataset in ('train', 'validation', 'test'):
        os.makedirs(dirs['data_info'], exist_ok=True)
        desc_path = os.path.join(dirs['data_info'],
                                 'descr_{}_{}_{}.ods'.format(args['task'], args['image_type'], dataset))
        with pd.ExcelWriter(desc_path, mode='w') as writer:
            for feature in ['sex', 'age_approx', 'location']:
                logs_df = df[['image_type', 'class', feature]].value_counts(sort=False, dropna=False).to_frame('counts')
                logs_df = logs_df.pivot_table(values='counts', fill_value=0,
                                              index=['image_type', feature], columns='class', aggfunc=sum)
                logs_df = logs_df[class_names]  # ?
                logs_df.to_excel(writer, sheet_name=feature)
