import os
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from features_def import BEN_MAL_MAP, LOCATIONS, IMAGE_TYPE, SEX, AGE_APPROX, TASK_CLASSES
from settings import data_csv


def read_image(sample):
    sample['image'] = tf.cast(x=tf.io.decode_image(tf.io.read_file(sample['image_path']), channels=3),
                              dtype=tf.float32)
    return sample


class MelData:
    def __init__(self, args, dirs):
        self.args = args
        self.dirs = dirs
        self.rng = tf.random.Generator.from_non_deterministic_state()  # .from_seed(1312)
        self.class_names = TASK_CLASSES[self.args['task']]

    def prep_df(self, mode: str):
        df = pd.read_csv(data_csv[mode])
        if mode in ('train', 'validation'):
            df = df.sample(frac=self.args['dataset_frac'], random_state=1312)
        df['image'] = df['image'].apply(lambda x: os.path.join(self.dirs['proc_img_folder'], x))
        df['location'].fillna('', inplace=True)
        df['sex'].fillna('', inplace=True)
        df['age_approx'].fillna(-1, inplace=True)
        df['age_approx'] = df['age_approx'].astype(int).astype('string')
        df['image_type'].fillna('', inplace=True)
        if mode != 'isic20_test':
            if self.args['task'] == 'ben_mal':
                df.replace(to_replace=BEN_MAL_MAP, inplace=True)
            elif self.args['task'] in ('nev_mel', '5cls'):
                if self.args['task'] == 'nev_mel':
                    df.drop(df[~df['class'].isin(self.class_names)].index, errors='ignore', inplace=True)
                df.drop(df[df['class'].isin(['UNK'])].index, errors='ignore', inplace=True)
        if mode == 'validation' and self.args['clinic_val']:
            df.drop(df[~df['image_type'].isin(['clinic'])].index, errors='ignore', inplace=True)
        elif self.args['image_type'] != 'both':  # Keep derm or clinic, samples.
            df.drop(df[~df['image_type'].isin([self.args['image_type']])].index, errors='ignore', inplace=True)
        # log datasets
        if mode in ('train', 'validation', 'test'):
            os.makedirs(self.dirs['data_info'], exist_ok=True)
            desc_path = os.path.join(self.dirs['data_info'],
                                     'descr_{}_{}_{}.ods'.format(self.args['task'], self.args['image_type'], mode))
            with pd.ExcelWriter(desc_path, mode='w') as writer:
                for feature in ['sex', 'age_approx', 'location']:
                    logs_df = df[['image_type', 'class', feature]].value_counts(sort=False,
                                                                                dropna=False).to_frame('counts')
                    logs_df = logs_df.pivot_table(values='counts', fill_value=0,
                                                  index=['image_type', feature], columns='class', aggfunc=sum)
                    logs_df = logs_df[self.class_names]
                    logs_df.to_excel(writer, sheet_name=feature)

        onehot_input = {'image_path': df['image'].values}
        img_type_lookup = tf.keras.layers.StringLookup(vocabulary=IMAGE_TYPE, output_mode='one_hot')
        loc_lookup = tf.keras.layers.StringLookup(vocabulary=LOCATIONS, output_mode='one_hot')
        sex_lookup = tf.keras.layers.StringLookup(vocabulary=SEX, output_mode='one_hot')
        age_lookup = tf.keras.layers.StringLookup(vocabulary=AGE_APPROX, output_mode='one_hot')

        onehot_input['anatom_site_general'] = loc_lookup(tf.convert_to_tensor(df['location'].values))[:, 1:]  # compat
        onehot_input['location'] = loc_lookup(tf.convert_to_tensor(df['location'].values))[:, 1:]
        onehot_input['sex'] = sex_lookup(tf.convert_to_tensor(df['sex'].values))[:, 1:]
        onehot_input['age_approx'] = age_lookup(tf.convert_to_tensor(df['age_approx'].values))[:, 1:]
        onehot_input['image_type'] = img_type_lookup(tf.convert_to_tensor(df['image_type'].values))[:, 1:]

        if not self.args['no_clinical_data']:
            onehot_input['clinical_data'] = tf.keras.layers.Concatenate()([onehot_input['location'],
                                                                           onehot_input['sex'],
                                                                           onehot_input['age_approx']])
            if not self.args['no_image_type']:
                onehot_input['clinical_data'] = tf.keras.layers.Concatenate()([onehot_input['clinical_data'],
                                                                               onehot_input['image_type']])

        onehot_label = None
        if mode != 'isic20_test':
            label_lookup = tf.keras.layers.StringLookup(vocabulary=self.class_names, output_mode='one_hot')
            onehot_label = {'class': label_lookup(df['class'].values)[:, 1:]}

        sample_weights = None
        if mode == 'train':
            if self.args['image_type'] == 'both' and self.args['weighted_samples']:  # Sample weight for image type
                # sample_weights = tf.math.divide(tf.reduce_sum(onehot_input['image_type']),
                #                                 tf.math.multiply(tf.cast(2., dtype=tf.float32),
                #                                                  tf.reduce_sum(onehot_input['image_type'], axis=0)))
                samples_per_image_type = tf.reduce_sum(onehot_input['image_type'], axis=0)
                sample_weights = tf.math.divide(tf.reduce_max(samples_per_image_type), samples_per_image_type)
                sample_weights = tf.gather(sample_weights, tf.math.argmax(onehot_input['image_type'], axis=-1))
            if self.args['weighted_loss']:  # Class weight
                # class_weights = tf.math.divide(tf.reduce_sum(onehot_label['class']),
                #                                tf.math.multiply(tf.cast(len(self.class_names), dtype=tf.float32),
                #                                                 tf.reduce_sum(onehot_label['class'], axis=0)))
                samples_per_class = tf.reduce_sum(onehot_label['class'], axis=0)
                class_weights = tf.math.divide(tf.reduce_max(samples_per_class), samples_per_class)
                class_weights = tf.gather(class_weights, tf.math.argmax(onehot_label['class'], axis=-1))

                if sample_weights is not None:  # `class_weight` and `sample_weight` are multiplicative.
                    class_weights = tf.cast(class_weights, sample_weights.dtype)
                    sample_weights = sample_weights * class_weights
                else:
                    sample_weights = class_weights
            if sample_weights is None:
                sample_weights = tf.ones(onehot_label['class'].shape[0], dtype=tf.float32)
        return onehot_input, onehot_label, sample_weights

    def get_dataset(self, dataset_name=None):
        data = self.prep_df(dataset_name)
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if dataset_name == 'train':
            dataset = dataset.shuffle(buffer_size=dataset.cardinality(), reshuffle_each_iteration=True)
        dataset = dataset.map(lambda sample, label, sample_weights:
                              (read_image(sample=sample), label, sample_weights),
                              num_parallel_calls=16)
        dataset = dataset.batch(self.args['batch_size'] * self.args['gpus'])
        if dataset_name == 'train':
            dataset = dataset.map(lambda sample, label, sample_weights:
                                  (self.augm(sample), label, sample_weights),
                                  num_parallel_calls=16)
        elif dataset_name == 'isic20_test':
            dataset = dataset.map(lambda sample, label, sample_weights: sample,
                                  num_parallel_calls=16)
        else:
            dataset = dataset.map(lambda sample, label, sample_weights:
                                  (sample, label),
                                  num_parallel_calls=16)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        if dataset_name != 'train':
            dataset = dataset.repeat(1)
        return dataset.prefetch(buffer_size=16)

    def augm(self, sample):
        image = tf.image.random_flip_up_down(image=sample['image'])
        image = tf.image.random_flip_left_right(image=image)
        image = tf.image.random_brightness(image=image, max_delta=60.)
        image = tf.image.random_contrast(image=image, lower=.5, upper=1.5)
        image = tf.clip_by_value(image, clip_value_min=0., clip_value_max=255.)
        image = tf.image.random_saturation(image=image, lower=0.8, upper=1.2)
        image = tfa.image.sharpness(image=image,
                                    factor=self.rng.uniform(shape=[1], minval=0.5, maxval=1.5),
                                    name='Sharpness')  # _sharpness_image -> image_channels = tf.shape(image)[-1]
        trans_val = self.args['image_size'] * 0.2
        image = tfa.image.translate(images=image,
                                    translations=self.rng.uniform(shape=[2], minval=-trans_val, maxval=trans_val,
                                                                  dtype=tf.float32), name='Translation')
        image = tfa.image.rotate(images=image,
                                 angles=tf.cast(self.rng.uniform(shape=[], minval=0, maxval=360, dtype=tf.int32),
                                                dtype=tf.float32),
                                 interpolation='bilinear', name='Rotation')
        image = tf.cond(tf.less(self.rng.uniform(shape=[1]), 0.5),
                        lambda: tfa.image.gaussian_filter2d(image=image,
                                                            sigma=1.5, filter_shape=3, name='Gaussian_filter'),
                        lambda: image)

        # sample['image'] = tf.map_fn(fn=_augm, elems=sample['image'], fn_output_signature=tf.float32)
        cutout_ratio = 0.15
        for i in range(3):
            mask_height = tf.cast(self.rng.uniform(shape=[], minval=0, maxval=self.args['image_size'] * cutout_ratio),
                                  dtype=tf.int32) * 2
            mask_width = tf.cast(self.rng.uniform(shape=[], minval=0, maxval=self.args['image_size'] * cutout_ratio),
                                 dtype=tf.int32) * 2
            image = tfa.image.random_cutout(image, mask_size=(mask_height, mask_width))
        sample['image'] = \
            {'xept': tf.keras.applications.xception.preprocess_input,
             'incept': tf.keras.applications.inception_v3.preprocess_input,
             'effnet0': tf.keras.applications.efficientnet.preprocess_input,
             'effnet1': tf.keras.applications.efficientnet.preprocess_input,
             'effnet6': tf.keras.applications.efficientnet.preprocess_input
             }[self.args['pretrained']](image)
        return sample
