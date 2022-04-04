import os
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from config import BEN_MAL_MAP, LOCATIONS, IMAGE_TYPE, SEX, AGE_APPROX, TASK_CLASSES, \
    MAIN_DIR, ISIC20_TEST_PATH, TEST_CSV_PATH, VAL_CSV_PATH, TRAIN_CSV_PATH, ISIC16_TEST_PATH, \
    ISIC17_TEST_PATH, ISIC18_VAL_TEST_PATH, DERMOFIT_TEST_PATH, UP_TEST_PATH, MCLASS_CLINIC_TEST_PATH, \
    MCLASS_DERM_TEST_PATH


class MelData:
    def __init__(self, args):
        self.task = args['task']
        self.image_type = args['image_type']
        self.pretrained = args['pretrained']
        self.input_shape = args['input_shape']
        self.no_clinical_data = args['no_clinical_data']
        self.no_image_type = args['no_image_type']
        self.TF_RNG = tf.random.Generator.from_non_deterministic_state()  # .from_seed(1312)
        self.class_names = TASK_CLASSES[self.task]
        self.weighted_samples = args['weighted_samples']
        self.weighted_loss = args['weighted_loss']
        self.num_classes = len(self.class_names)
        self.dir_dict = args['dir_dict']
        self.dataset_frac = args['dataset_frac']
        self.batch_size = args['batch_size']
        # 'train', 'validation', 'test', 'isic16_test', 'isic17_test', 'isic20_test', 'isic18_val_test', 'dermofit_test', 'up_test'
                        # 'validation': self.prep_df(mode='validation').sample(frac=dataset_frac, random_state=1312)}

        # self.train_len = len(self.data_df['train'])

    def prep_df(self, mode: str):
        df = pd.read_csv({'train': TRAIN_CSV_PATH, 'validation': VAL_CSV_PATH, 'test': TEST_CSV_PATH,
                          'isic20_test': ISIC20_TEST_PATH, 'isic16_test': ISIC16_TEST_PATH,
                          'isic17_test': ISIC17_TEST_PATH, 'isic18_val_test': ISIC18_VAL_TEST_PATH,
                         'dermofit_test': DERMOFIT_TEST_PATH, 'up_test': UP_TEST_PATH,
                          'mclass_clinic_test': MCLASS_CLINIC_TEST_PATH,
                          'mclass_derm_test': MCLASS_DERM_TEST_PATH}[mode])
        if mode in ('train', 'validation'):
            df = df.sample(frac=self.dataset_frac, random_state=1312)
        df['image'] = df['image'].apply(lambda x: os.path.join(self.dir_dict['data_folder'], x))
        df['location'].fillna('', inplace=True)
        df['sex'].fillna('', inplace=True)
        df['age_approx'].fillna(-1, inplace=True)
        df['age_approx'] = df['age_approx'].astype(int).astype('string')
        df['image_type'].fillna('', inplace=True)
        if mode != 'isic20_test':
            if self.task == 'ben_mal':
                df.replace(to_replace=BEN_MAL_MAP, inplace=True)
            elif self.task in ('nev_mel', '5cls'):
                if self.task == 'nev_mel':
                    df.drop(df[~df['class'].isin(self.class_names)].index, errors='ignore', inplace=True)
                df.drop(df[df['class'].isin(['UNK'])].index, errors='ignore', inplace=True)
        if self.image_type != 'both':  # Keep derm or clinic, samples.
            df.drop(df[~df['image_type'].isin([self.image_type])].index, errors='ignore', inplace=True)
        #log datasets
        if mode in ('train', 'validation', 'test'):
            os.makedirs(os.path.join(MAIN_DIR, 'data_info'), exist_ok=True)
            with pd.ExcelWriter(os.path.join(MAIN_DIR, 'data_info', 'descr_{}_{}_{}.xlsx'.format(self.task, self.image_type, mode)), mode='w') as writer:
                for feature in ['sex', 'age_approx', 'location']:
                    logs_df = df[['image_type', 'class', feature]].value_counts(sort=False, dropna=False).to_frame('counts')
                    logs_df = logs_df.pivot_table(values='counts', fill_value=0, index=['image_type', feature], columns='class', aggfunc=sum)
                    logs_df = logs_df[self.class_names]
                    logs_df.to_excel(writer, sheet_name=feature)
        # oversampling
        # if mode == 'train':
        #     max_size = max(df[['image_type', 'class']].value_counts(sort=False, dropna=False))
        #     list_of_sub_df = []
        #     for _image_type in df['image_type'].unique():
        #         for _class in df['class'].unique():
        #             sub_df = df.loc[(df['image_type'] == _image_type) & (df['class'] == _class)]
        #             sub_df = pd.concat([sub_df] * (max_size // len(sub_df)))
        #             sub_df = pd.concat([sub_df, sub_df.sample(n=max_size - len(sub_df))])
        #             list_of_sub_df.append(sub_df)
        #     df = pd.concat(list_of_sub_df)

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

        if not self.no_clinical_data:
            onehot_input['clinical_data'] = tf.keras.layers.Concatenate()([onehot_input['location'], onehot_input['sex'], onehot_input['age_approx']])
            if not self.no_image_type:
                onehot_input['clinical_data'] = tf.keras.layers.Concatenate()([onehot_input['clinical_data'], onehot_input['image_type']])

        onehot_label = None
        if mode != 'isic20_test':
            label_lookup = tf.keras.layers.StringLookup(vocabulary=self.class_names, output_mode='one_hot')
            onehot_label = {'class': label_lookup(df['class'].values)[:, 1:]}

        sample_weights = None
        if mode == 'train':
            if self.image_type == 'both' and self.weighted_samples:  # Sample weight for image type
                sample_weights = tf.math.divide(tf.reduce_sum(onehot_input['image_type']),
                                                tf.math.multiply(tf.cast(onehot_input['image_type'].shape[-1], dtype=tf.float32),
                                                                 tf.reduce_sum(onehot_input['image_type'], axis=0)))
                sample_weights = tf.gather(sample_weights, tf.math.argmax(onehot_input['image_type'], axis=-1))
            if self.weighted_loss:  # Class weight
                class_weights = tf.math.divide(tf.reduce_sum(onehot_label['class']),
                                               tf.math.multiply(tf.cast(self.num_classes, dtype=tf.float32),
                                                                tf.reduce_sum(onehot_label['class'], axis=0)))
                class_weights = tf.gather(class_weights, tf.math.argmax(onehot_label['class'], axis=-1))

                if sample_weights is not None: # `class_weight` and `sample_weight` are multiplicative.
                    class_weights = tf.cast(class_weights, sample_weights.dtype)
                    sample_weights = sample_weights * class_weights
                else:
                    sample_weights = class_weights
            if sample_weights is None:
                sample_weights = tf.ones(onehot_label['class'].shape[0], dtype=tf.float32)
        return onehot_input, onehot_label, sample_weights

    def get_dataset(self, mode=None):
        data = self.prep_df(mode)
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.map(lambda sample, label, sample_weights: (self.read_image(sample=sample), label, sample_weights), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        if mode != 'train':
            dataset = dataset.map(lambda sample, label, sample_weights: (self.augm(sample), label), num_parallel_calls=tf.data.AUTOTUNE)
        if mode == 'isic20_test':
            dataset = dataset.map(lambda sample, label, sample_weights: sample, num_parallel_calls=tf.data.AUTOTUNE)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)

        if mode != 'train':
            dataset = dataset.repeat(1)
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def read_image(self, sample):
        sample['image'] = tf.cast(tf.io.decode_image(tf.io.read_file(sample['image_path']), channels=3), dtype=tf.float32)
        return sample

    def augm(self, sample):
        image = tf.image.random_flip_up_down(image=sample['image'])
        image = tf.image.random_flip_left_right(image=image)
        image = tf.image.random_brightness(image=image, max_delta=60.)
        image = tf.image.random_contrast(image=image, lower=.5, upper=1.5)
        image = tf.clip_by_value(image, clip_value_min=0., clip_value_max=255.)
        image = tf.image.random_saturation(image=image, lower=0.8, upper=1.2)
        image = tfa.image.sharpness(image=image, factor=self.TF_RNG.uniform(shape=[1], minval=0.5, maxval=1.5), name='Sharpness') # _sharpness_image ->     image_channels = tf.shape(image)[-1]
        trans_val = self.input_shape[0] * 0.2
        image = tfa.image.translate(images=image, translations=self.TF_RNG.uniform(shape=[2], minval=-trans_val, maxval=trans_val,
                                                                                   dtype=tf.float32), name='Translation')
        image = tfa.image.rotate(images=image, angles=tf.cast(self.TF_RNG.uniform(shape=[], minval=0, maxval=360,
                                                                                  dtype=tf.int32), dtype=tf.float32), interpolation='bilinear', name='Rotation')
        image = tf.cond(tf.less(self.TF_RNG.uniform(shape=[1]), 0.5),
                        lambda: tfa.image.gaussian_filter2d(image=image, sigma=1.5, filter_shape=3, name='Gaussian_filter'),
                        lambda: image)

        # sample['image'] = tf.map_fn(fn=_augm, elems=sample['image'], fn_output_signature=tf.float32)
        cutout_ratio = 0.15
        for i in range(3):
            mask_height = tf.cast(self.TF_RNG.uniform(shape=[], minval=0, maxval=self.input_shape[0] * cutout_ratio),
                                  dtype=tf.int32) * 2
            mask_width = tf.cast(self.TF_RNG.uniform(shape=[], minval=0, maxval=self.input_shape[0] * cutout_ratio),
                                 dtype=tf.int32) * 2
            image = tfa.image.random_cutout(image, mask_size=(mask_height, mask_width))
        sample['image'] = {'xept': tf.keras.applications.xception.preprocess_input,
                           'incept': tf.keras.applications.inception_v3.preprocess_input,
                           'effnet0': tf.keras.applications.efficientnet.preprocess_input,
                           'effnet1': tf.keras.applications.efficientnet.preprocess_input,
                           'effnet6': tf.keras.applications.efficientnet.preprocess_input}[self.pretrained](image)
        return sample
