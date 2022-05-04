import os
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from features_def import BEN_MAL_MAP, LOCATIONS, IMAGE_TYPE, SEX, AGE_APPROX, TASK_CLASSES
from settings import data_csv


def read_image(sample):
    sample['image'] = tf.cast(x=tf.io.decode_image(tf.io.read_file(sample['image_path']), channels=3), dtype=tf.float32)
    return sample


class MelData:
    def __init__(self, args, dirs):
        self.args = args
        self.dirs = dirs
        self.rng = tf.random.Generator.from_non_deterministic_state()  # .from_seed(1312)
        self.class_names = TASK_CLASSES[self.args['task']]

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

    def _df_to_onehot_inputs_dict(self, df):
        onehot_input_dict = {'image_path': df['image'].values}
        for key, voc in (('location', LOCATIONS), ('sex', SEX), ('age_approx', AGE_APPROX), ('image_type', IMAGE_TYPE)):
            lookup = tf.keras.layers.StringLookup(vocabulary=voc, output_mode='one_hot')
            onehot_input_dict[key] = lookup(tf.convert_to_tensor(df[key].values))[:, 1:]
        onehot_input_dict['anatom_site_general'] = onehot_input_dict['location']  # compat

        if not self.args['no_clinical_data']:
            onehot_input_dict['clinical_data'] = tf.concat([onehot_input_dict['location'], onehot_input_dict['sex'],
                                                       onehot_input_dict['age_approx']])
            if not self.args['no_image_type']:
                onehot_input_dict['clinical_data'] = tf.concat([onehot_input_dict['clinical_data'], onehot_input_dict['image_type']])
        return onehot_input_dict

    def prep_df(self, mode: str):
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

        self._log_info(mode=mode, df=df)

        onehot_input_dict = self._df_to_onehot_inputs_dict(df)
        onehot_label = None
        sample_weight = None

        if 'class' in df.columns:  # If class available, make onehot
            onehot_label = {'class': tf.keras.layers.StringLookup(vocabulary=self.class_names,
                                                                  output_mode='one_hot')(df['class'].values)[:, 1:]}

        if mode == 'train':
            if self.args['image_type'] == 'both' and self.args['weighted_samples']:  # Sample weight for image type
                samples_per_image_type = tf.reduce_sum(onehot_input_dict['image_type'], axis=0)
                sample_weight = tf.math.divide(tf.reduce_max(samples_per_image_type), samples_per_image_type)
                sample_weight = tf.gather(sample_weight, tf.math.argmax(onehot_input_dict['image_type'], axis=-1))

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

    def get_dataset(self, dataset=None):
        ds = tf.data.Dataset.from_tensor_slices(self.prep_df(dataset))
        if dataset == 'train':
            ds = ds.shuffle(buffer_size=ds.cardinality(), reshuffle_each_iteration=True)
        # Read image
        ds = ds.map(lambda sample, label, sample_weights: (read_image(sample=sample), label, sample_weights),
                    num_parallel_calls=16)
        ds = ds.batch(self.args['batch_size'] * self.args['gpus'])  # Batch samples
        if dataset == 'train':  # Apply image data augmentation on training dataset
            ds = ds.map(lambda sample, label, sample_weights: (self.augm(sample), label, sample_weights),
                        num_parallel_calls=16)
        else:  # Remove sample_weights from validation and test datasets
            ds = ds.map(lambda sample, label, sample_weights: (sample, label),
                        num_parallel_calls=16)
        if dataset == 'isic20_test':  # Remove label when isic20_test is running
            ds = ds.map(lambda sample, label: sample,
                        num_parallel_calls=16)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)
        if dataset != 'train':
            ds = ds.repeat(1)
        return ds.prefetch(buffer_size=64)

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
