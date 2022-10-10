import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from features_def import BEN_MAL_MAP, LOCATIONS, IMAGE_TYPE, SEX, AGE_APPROX, TASK_CLASSES
from settings import data_csv


def _prep_df(args, dataset: str, dirs):
    df = pd.read_csv(data_csv[dataset])
    # _log_info(args, dataset, df, dirs)

    class_names = TASK_CLASSES[args['task']]

    if float(args['dataset_frac']) != 1.:
        if dataset in ('train', 'validation'):
            df = df.sample(frac=args['dataset_frac'], random_state=1312)
    else:
        if dataset in 'train':
            df = df.sample(frac=args['dataset_frac'], random_state=1312)

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
            df = df.drop(df[~df['class'].isin(class_names)].index, errors='ignore')
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
    from sklearn.preprocessing import OneHotEncoder
    df = _prep_df(args, dataset, dirs)
    onehot_feature_dict = {'image_path': df['image'].values}
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
    if not args['no_clinical_data']:
        onehot_feature_dict['clinical_data'] = ohe_data[:, :-2]
    sample_weight = None
    onehot_label = {'class': ohe_data[:, -2:]}

    if dataset == 'train':
        if args['image_type'] == 'both' and args['weighted_samples']:  # Sample weight for image type
            image_type_ohe = ohe_data[:, -4:-2]
            sample_weight = np.divide(np.amax(np.sum(image_type_ohe, axis=0)), np.sum(image_type_ohe, axis=0))
            sample_weight = np.sum(np.multiply(sample_weight, image_type_ohe), axis=-1)
        if args['weighted_loss']:  # Class weight
            class_weight = np.divide(np.amax(np.sum(onehot_label['class'], axis=0)), np.sum(onehot_label['class'], axis=0))
            class_weight = np.sum(np.multiply(class_weight, onehot_label['class']), axis=-1)

            if sample_weight is not None:  # From keras: `class_weight` and `sample_weight` are multiplicative.
                sample_weight = sample_weight * class_weight
            else:
                sample_weight = class_weight

        if sample_weight is None:  # Set sample weight to one if not set.
            sample_weight = np.ones(len(df), dtype=tf.float32)
    return onehot_feature_dict, onehot_label, sample_weight


def _read_image(sample):
    read_img = tf.io.read_file(sample['image_path'])
    decode_img = tf.io.decode_image(read_img, channels=3)
    sample['image'] = tf.cast(x=decode_img, dtype=tf.float32)
    del read_img
    del decode_img
    return sample


def get_dataset(args, dataset, dirs):
    rng = tf.random.Generator.from_non_deterministic_state()
    ds = tf.data.Dataset.from_tensor_slices(_prep_df_for_tfdataset(args, dataset, dirs))

    # Memory leak due to shuffle: https://github.com/tensorflow/tensorflow/issues/44176#issuecomment-783768033
    # if dataset == 'train':
    #     ds = ds.shuffle(buffer_size=ds.cardinality(), seed=1312, reshuffle_each_iteration=True)
    # Read image
    # ds = ds.map(lambda sample, label, sample_weights: (_read_image(sample=sample), label, sample_weights), num_parallel_calls=tf.data.AUTOTUNE)
    if dataset == 'train':  # Apply image data augmentation on training dataset
        ds = ds.map(lambda sample, label, sample_weights: (augm(_read_image(sample), args, rng), label, sample_weights), num_parallel_calls=tf.data.AUTOTUNE)
    elif dataset == 'isic20_test':  # Remove sample_weights from validation and test datasets
        ds = ds.map(lambda sample, label, sample_weights: _read_image(sample), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda sample, label, sample_weights: (_read_image(sample), label), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(args['batch_size'] * args['gpus'])  # Batch samples
    # ds = ds.repeat(1)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    return ds.prefetch(8)  # tf.data.AUTOTUNE)  # buffer_size=10 * args['batch_size'] * args['gpus'])


def augm(sample, args, rng):
    img = tf.image.random_flip_up_down(tf.image.random_flip_left_right(image=sample['image']))
    img = tf.image.random_brightness(image=tf.image.random_contrast(image=img, lower=.5, upper=1.5), max_delta=60.)
    img = tf.image.random_saturation(image=tf.clip_by_value(img, clip_value_min=0., clip_value_max=255.), lower=0.8, upper=1.2)
    # _sharpness_image -> image_channels = tf.shape(image)[-1]
    img = tfa.image.sharpness(image=img, factor=rng.uniform(shape=[1], minval=0.5, maxval=1.5), name='Sharpness')
    img = tfa.image.translate(images=img, translations=rng.uniform(shape=[2],
                                                                   minval=-args['image_size'] * 0.2,
                                                                   maxval=args['image_size'] * 0.2,
                                                                   dtype=tf.float32), name='Translation')
    img = tfa.image.rotate(images=img, angles=tf.cast(rng.uniform(shape=[],  dtype=tf.int32,
                                                                  minval=0, maxval=360), dtype=tf.float32),
                           interpolation='bilinear', name='Rotation')
    img = tf.cond(tf.less(rng.uniform(shape=[1]), 0.5),
                  lambda: tfa.image.gaussian_filter2d(image=img, sigma=1.5, filter_shape=3, name='Gaussian_filter'),
                  lambda: img)
    cutout_ratio = 0.15
    img = tf.expand_dims(img, 0)
    for i in range(3):
        mask_height = tf.cast(rng.uniform(shape=[], minval=0, maxval=args['image_size'] * cutout_ratio),
                              dtype=tf.int32) * 2
        mask_width = tf.cast(rng.uniform(shape=[], minval=0, maxval=args['image_size'] * cutout_ratio),
                             dtype=tf.int32) * 2
        img = tfa.image.random_cutout(img, mask_size=(mask_height, mask_width))
    img = tf.squeeze(img)
    sample['image'] = {'xept': tf.keras.applications.xception.preprocess_input,
                       'incept': tf.keras.applications.inception_v3.preprocess_input,
                       'effnet0': tf.keras.applications.efficientnet.preprocess_input,
                       'effnet1': tf.keras.applications.efficientnet.preprocess_input,
                       'effnet6': tf.keras.applications.efficientnet.preprocess_input
                       }[args['pretrained']](img)
    return sample


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
