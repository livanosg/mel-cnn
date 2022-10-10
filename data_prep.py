import os
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from features_def import BEN_MAL_MAP, LOCATIONS, IMAGE_TYPE, SEX, AGE_APPROX, TASK_CLASSES
from settings import data_csv


class MelData:
    def __init__(self, args, dirs):
        self.args = args
        self.dirs = dirs
        self.rng = tf.random.Generator.from_non_deterministic_state()  # .from_seed(1312)
        self.class_names = TASK_CLASSES[self.args['task']]
        self.dfs = {key: self._prep_df(key) for key in data_csv.keys()}
        [self._log_info(mode=mode, df=self.dfs[mode]) for mode in ('train', 'validation', 'test')]

    def _log_info(self, mode, df):
        """log datasets general information"""
        if mode in ('train', 'validation', 'test'):
            os.makedirs(self.dirs['data_info'], exist_ok=True)
            desc_path = os.path.join(self.dirs['data_info'],
                                     'descr_{}_{}_{}.ods'.format(self.args['task'], self.args['image_type'], mode))
            with pd.ExcelWriter(desc_path, mode='w') as writer:
                for feature in ['sex', 'age_approx', 'location']:
                    logs_df = df[['image_type', 'class', feature]].value_counts(sort=False, dropna=False).to_frame('counts')
                    logs_df = logs_df.pivot_table(values='counts', fill_value=0,
                                                  index=['image_type', feature], columns='class', aggfunc=sum)
                    logs_df = logs_df[self.class_names]  # ?
                    logs_df.to_excel(writer, sheet_name=feature)

    def _prep_df(self, mode: str):
        df = pd.read_csv(data_csv[mode])
        if mode in ('train', 'validation'):
            df = df.sample(frac=self.args['dataset_frac'], random_state=1312)
        # Set proper folder to fetch images
        df['image'] = df['image'].apply(lambda x: os.path.join(self.dirs['proc_img_folder'], x))
        # Handle NaNs
        for cat in ['location', 'sex', 'image_type']:
            df[cat].fillna('', inplace=True)
        df['age_approx'].fillna(-1, inplace=True)
        df['age_approx'] = df['age_approx'].astype(int).astype('string')

        # Define classes according to task
        if 'class' in df.columns:
            if self.args['task'] == 'ben_mal':
                df.replace(to_replace=BEN_MAL_MAP, inplace=True)
            if self.args['task'] == 'nev_mel':
                df.drop(df[~df['class'].isin(self.class_names)].index, errors='ignore', inplace=True)
            if self.args['task'] == '5cls':  # Drop unclassified benign samples
                df.drop(df[df['class'].isin(['UNK'])].index, errors='ignore', inplace=True)

        if mode == 'validation':
            if self.args['clinic_val']:  # Run validation on clinical dataset regardless the training image type
                df.drop(df[df['image_type'].isin(['derm'])].index, errors='ignore', inplace=True)
            elif self.args['image_type'] != 'both':  # Run validation on the same training image type
                df.drop(df[~df['image_type'].isin([self.args['image_type']])].index, errors='ignore', inplace=True)
        else:  # Keep dermoscopy or clinical image samples for the rest datasets according to training image type
            if self.args['image_type'] != 'both':
                df.drop(df[~df['image_type'].isin([self.args['image_type']])].index, errors='ignore', inplace=True)
        return df

    def _prep_df_for_tf_dataset(self, mode):
        df = self.dfs[mode]
        onehot_input_dict = {'image_path': df['image'].values}
        onehot_feature_dict = {}
        for key, voc in (('location', LOCATIONS), ('sex', SEX), ('age_approx', AGE_APPROX), ('image_type', IMAGE_TYPE)):
            lookup = tf.keras.layers.StringLookup(vocabulary=voc, output_mode='one_hot')
            onehot_feature_dict[key] = lookup(tf.convert_to_tensor(df[key].values))[:, 1:]
        onehot_feature_dict['anatom_site_general'] = onehot_feature_dict['location']  # compat

        if not self.args['no_clinical_data']:
            onehot_input_dict['clinical_data'] = tf.keras.layers.Concatenate()([onehot_feature_dict['location'],
                                                                                onehot_feature_dict['sex'],
                                                                                onehot_feature_dict['age_approx']])
            if not self.args['no_image_type']:
                onehot_input_dict['clinical_data'] = tf.keras.layers.Concatenate()([onehot_input_dict['clinical_data'],
                                                                                    onehot_feature_dict['image_type']])
        onehot_label = None
        sample_weight = None

        if 'class' in df.columns:  # If class available, make onehot
            onehot_label = {'class': tf.keras.layers.StringLookup(vocabulary=self.class_names,
                                                                  output_mode='one_hot')(df['class'].values)[:, 1:]}

        if mode == 'train':
            if self.args['image_type'] == 'both' and self.args['weighted_samples']:  # Sample weight for image type
                samples_per_image_type = tf.reduce_sum(onehot_feature_dict['image_type'], axis=0)
                sample_weight = tf.math.divide(tf.reduce_max(samples_per_image_type), samples_per_image_type)
                sample_weight = tf.gather(sample_weight, tf.math.argmax(onehot_feature_dict['image_type'], axis=-1))

            if self.args['weighted_loss']:  # Class weight
                samples_per_class = tf.reduce_sum(onehot_label['class'], axis=0)
                class_weight = tf.math.divide(tf.reduce_max(samples_per_class), samples_per_class)
                class_weight = tf.gather(class_weight, tf.math.argmax(onehot_label['class'], axis=-1))

                if sample_weight is not None:  # From keras: `class_weight` and `sample_weight` are multiplicative.
                    class_weight = tf.cast(class_weight, sample_weight.dtype)
                    sample_weight = sample_weight * class_weight
                else:
                    sample_weight = class_weight

            if sample_weight is None:  # Set sample weight to one if not set.
                sample_weight = tf.ones(onehot_label['class'].shape[0], dtype=tf.float32)
        return onehot_input_dict, onehot_label, sample_weight

    def _read_image(self, sample):
        sample['image'] = tf.cast(x=tf.io.decode_image(tf.io.read_file(sample['image_path']), channels=3),
                                  dtype=tf.float32)
        return sample

    def get_dataset(self, dataset=None):
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = tf.data.Dataset.from_tensor_slices(self._prep_df_for_tf_dataset(mode=dataset))
        ds = ds.with_options(options)

        # Memory leak due to shuffle: https://github.com/tensorflow/tensorflow/issues/44176#issuecomment-783768033
        # if dataset == 'train':
        #     buffer_size = ds.cardinality()
        #     ds = ds.shuffle(buffer_size=buffer_size, seed=1312, reshuffle_each_iteration=True)
        # if dataset != 'train':
        # ds = ds.repeat(1)
        # Read image
        ds = ds.map(lambda sample, label, sample_weights: (self._read_image(sample=sample), label, sample_weights), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.args['batch_size'] * self.args['gpus'])  # Batch samples

        if dataset == 'train':  # Apply image data augmentation on training dataset
            ds = ds.map(lambda sample, label, sample_weights: (self.augm(sample), label, sample_weights), num_parallel_calls=tf.data.AUTOTUNE)
        elif dataset == 'isic20_test':  # Remove sample_weights from validation and test datasets
            ds = ds.map(lambda sample, label, sample_weights: sample, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(lambda sample, label, sample_weights: (sample, label), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(buffer_size=10 * self.args['batch_size'] * self.args['gpus'])

    def augm(self, sample):
        img = tf.image.random_flip_up_down(image=sample['image'])
        img = tf.image.random_flip_left_right(image=img)
        img = tf.image.random_brightness(image=img, max_delta=60.)
        img = tf.image.random_contrast(image=img, lower=.5, upper=1.5)
        img = tf.clip_by_value(img, clip_value_min=0., clip_value_max=255.)
        img = tf.image.random_saturation(image=img, lower=0.8, upper=1.2)
        img = tfa.image.sharpness(image=img, factor=self.rng.uniform(shape=[1], minval=0.5, maxval=1.5),
                                  name='Sharpness')  # _sharpness_image -> image_channels = tf.shape(image)[-1]
        trans_val = self.args['image_size'] * 0.2
        img = tfa.image.translate(images=img, translations=self.rng.uniform(shape=[2],
                                                                            minval=-trans_val, maxval=trans_val,
                                                                            dtype=tf.float32),
                                  name='Translation')
        img = tfa.image.rotate(images=img, angles=tf.cast(self.rng.uniform(shape=[],  dtype=tf.int32,
                                                                           minval=0, maxval=360), dtype=tf.float32),
                               interpolation='bilinear', name='Rotation')
        img = tf.cond(tf.less(self.rng.uniform(shape=[1]), 0.5),
                      lambda: tfa.image.gaussian_filter2d(image=img, sigma=1.5, filter_shape=3, name='Gaussian_filter'),
                      lambda: img)
        cutout_ratio = 0.15
        for i in range(3):
            mask_height = tf.cast(self.rng.uniform(shape=[], minval=0, maxval=self.args['image_size'] * cutout_ratio),
                                  dtype=tf.int32) * 2
            mask_width = tf.cast(self.rng.uniform(shape=[], minval=0, maxval=self.args['image_size'] * cutout_ratio),
                                 dtype=tf.int32) * 2
            img = tfa.image.random_cutout(img, mask_size=(mask_height, mask_width))
        sample['image'] = {'xept': tf.keras.applications.xception.preprocess_input,
                           'incept': tf.keras.applications.inception_v3.preprocess_input,
                           'effnet0': tf.keras.applications.efficientnet.preprocess_input,
                           'effnet1': tf.keras.applications.efficientnet.preprocess_input,
                           'effnet6': tf.keras.applications.efficientnet.preprocess_input
                           }[self.args['pretrained']](img)
        return sample
