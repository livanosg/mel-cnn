import os
import pandas as pd
import tensorflow as tf
from features_def import BEN_MAL_MAP, LOCATIONS, IMAGE_TYPE, SEX, AGE_APPROX, TASK_CLASSES
from settings import PROC_DATA_DIR, DATA_DIR


class MelData:
    def __init__(self, **kwargs):
        self.task = kwargs['task']
        self.mode = kwargs['mode']
        self.image_type = kwargs['image_type']

        self.rng = tf.random.Generator.from_non_deterministic_state()  # .from_seed(1312)
        self.class_names = TASK_CLASSES[kwargs['task']]
        self.csv_file = os.path.join(DATA_DIR, kwargs['mode'] + '.csv')

    @property
    def _prepare_df(self):
        """
        Read the CSV file, process image paths, handle missing values, and define classes according to the task.
        """
        df = pd.read_csv(self.csv_file)
        # Set proper folder to fetch images
        df['image'] = df['image'].apply(lambda x: os.path.join(PROC_DATA_DIR, x))
        # Handle NaNs
        for cat in ['location', 'sex', 'image_type']:
            df[cat].fillna('', inplace=True)
        df['age_approx'].fillna(-1, inplace=True)
        df['age_approx'] = df['age_approx'].astype(int).astype('string')

        # Define classes according to task
        if 'class' in df.columns:
            if self.task == 'ben_mal':
                df.replace(to_replace=BEN_MAL_MAP, inplace=True)
            if self.task == 'nev_mel':
                df.drop(~df.loc[self.class_names, 'class'].index, errors='ignore', inplace=True)
            if self.task == '5cls':  # Drop unclassified benign samples
                df.drop(df.loc['UNK', 'class'].index, errors='ignore', inplace=True)

        # Keep dermoscopy or clinical image samples for the rest datasets according to training image type
        if self.image_type != 'both':
            df.drop(~df.loc[self.image_type, 'image_type'].index, errors='ignore', inplace=True)
        return df

    def _prepare_tfds(self, get_clinical_features, get_image_type, get_class_weights, get_sample_weights, clinic_val):
        """
        Prepare TensorFlow datasets for training or evaluation.

        Args:
            get_clinical_features (bool): Whether to include clinical features in the dataset.
            get_image_type (bool): Whether to include image type in the dataset.
            get_class_weights (bool): Whether to calculate class weights for the dataset.
            get_sample_weights (bool): Whether to calculate sample weights for the dataset.

        Returns:
            tuple: A tuple containing the prepared features, labels, and sample weights.
                - features (dict): A dictionary containing the prepared image paths and clinical data.
                - labels (dict or None): A dictionary containing the one-hot encoded class labels, or None if class labels are not available.
                - sample_weights (tf.Tensor): A tensor containing the sample weights for training.

        """
        df = self._prepare_df
        if self.mode == 'val' and clinic_val:
            df.drop(df.loc['derm', 'image_type'].index, errors='ignore', inplace=True)

        onehot_feature_dict = {}
        for key, voc in (('location', LOCATIONS), ('sex', SEX), ('age_approx', AGE_APPROX), ('image_type', IMAGE_TYPE)):
            lookup = tf.keras.layers.StringLookup(vocabulary=voc, output_mode='one_hot')
            onehot_feature_dict[key] = lookup(tf.convert_to_tensor(df[key].values))[:, 1:]

        features = {'image_path': onehot_feature_dict['image']}
        if get_clinical_features:
            features['clinical_data'] = tf.keras.layers.Concatenate()([onehot_feature_dict['location'],
                                                                       onehot_feature_dict['sex'],
                                                                       onehot_feature_dict['age_approx']])
        if get_clinical_features and get_image_type:
            features['clinical_data'] = tf.keras.layers.Concatenate()([features['clinical_data'],
                                                                       onehot_feature_dict['image_type']])

        # Run validation on clinical dataset regardless the training image type

        labels = None
        if 'class' in self._prepare_df.columns:  # If class available, make onehot
            labels = {'class': tf.keras.layers.StringLookup(vocabulary=self.class_names,
                                                            output_mode='one_hot')(self._prepare_df['class'].values)[:,
                               1:]}

        sample_weights = tf.ones(labels['class'].shape[0], dtype=tf.float32)
        if self.mode == 'train':
            if get_image_type == 'both' and get_sample_weights:  # Sample weight for image type
                samples_per_image_type = tf.reduce_sum(onehot_feature_dict['image_type'], axis=0)
                sample_weights = tf.math.divide(tf.reduce_max(samples_per_image_type), samples_per_image_type)
                sample_weights = tf.gather(sample_weights, tf.math.argmax(onehot_feature_dict['image_type'], axis=-1))

            if get_class_weights:  # Class weight
                samples_per_class = tf.reduce_sum(labels['class'], axis=0)
                class_weight = tf.math.divide(tf.reduce_max(samples_per_class), samples_per_class)
                class_weight = tf.gather(class_weight, tf.math.argmax(labels['class'], axis=-1))

                if sample_weights:  # From keras: `class_weight` and `sample_weight` are multiplicative.
                    class_weight = tf.cast(class_weight, sample_weights.dtype)
                    sample_weights = sample_weights * class_weight
                else:
                    sample_weights = class_weight
        return features, labels, sample_weights

    def _read_image(self, sample):
        sample['image'] = tf.cast(x=tf.io.decode_image(tf.io.read_file(sample['image_path']), channels=3),
                                  dtype=tf.float32)
        return sample

    def get_dataset(self, get_clinical_features, get_image_type, get_class_weights, get_sample_weights, clinical_val):
        """
        Generates a TensorFlow dataset for training or evaluation.

        Args:
            get_clinical_features (bool): Whether to include clinical features in the dataset.
            get_image_type (bool): Whether to include image type in the dataset.
            get_class_weights (bool): Whether to calculate class weights for the dataset.
            get_sample_weights (bool): Whether to calculate sample weights for the dataset.

        Returns:
            tf.data.Dataset: The generated TensorFlow dataset.

        Raises:
            None
        """
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = tf.data.Dataset.from_tensor_slices(
            self._prepare_tfds(get_clinical_features, get_image_type, get_class_weights, get_sample_weights, clinical_val))
        ds = ds.with_options(options)

        ds = ds.map(lambda sample, label, sample_weights: (self._read_image(sample=sample), label, sample_weights),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.args['batch_size'] * self.args['gpus'])  # Batch samples

        if self.mode == 'train':  # Apply image data augmentation on training dataset
            ds = ds.map(lambda sample, label, sample_weights: (self.augm(sample), label, sample_weights),
                        num_parallel_calls=tf.data.AUTOTUNE)
        elif self.mode == 'isic20_test':  # Remove sample_weights from validation and test datasets
            ds = ds.map(lambda sample, label, sample_weights: sample, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(lambda sample, label, sample_weights: (sample, label), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(buffer_size=10 * self.args['batch_size'] * self.args['gpus'])

    def augm(self, sample):
        """
        Applies a series of image data augmentation techniques to the input sample.

        Parameters:
            sample (dict): A dictionary containing the input image and other sample-specific information.

        Returns:
            dict: The augmented sample dictionary.
        """
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
        img = tfa.image.rotate(images=img, angles=tf.cast(self.rng.uniform(shape=[], dtype=tf.int32,
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
