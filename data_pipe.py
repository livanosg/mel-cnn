import os
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications import xception, inception_v3, efficientnet
from config import BEN_MAL_MAP, LOCATIONS, IMAGE_TYPE, SEX, AGE_APPROX, TASK_CLASSES, MAIN_DIR, ISIC20_TEST_PATH, \
    TEST_CSV_PATH, VAL_CSV_PATH, TRAIN_CSV_PATH, ISIC16_TEST_PATH
from sklearn.preprocessing import OneHotEncoder


class MelData:
    def __init__(self, image_type: str, task: str, dir_dict: dict, dataset_frac: float, pretrained: str, input_shape: list):
        self.task = task
        self.image_type = image_type
        self.pretrained = pretrained
        self.input_shape = input_shape
        self.TF_RNG = tf.random.Generator.from_non_deterministic_state()  # .from_seed(1312)
        self.class_names = TASK_CLASSES[self.task]
        self.num_classes = len(self.class_names)
        self.dir_dict = dir_dict

        self.data_df = {'train': self.prep_df(mode='train').sample(frac=dataset_frac, random_state=1312),
                        'validation': self.prep_df(mode='validation').sample(frac=dataset_frac, random_state=1312),
                        'test': self.prep_df(mode='test'),
                        'isic16_test': self.prep_df(mode='isic16_test'),
                        'isic20_test': self.prep_df(mode='isic20_test')
                        }
        self.train_len = len(self.data_df['train'])

    def prep_df(self, mode: str):
        df = pd.read_csv({'train': TRAIN_CSV_PATH, 'validation': VAL_CSV_PATH, 'test': TEST_CSV_PATH,
                          'isic20_test': ISIC20_TEST_PATH, 'isic16_test': ISIC16_TEST_PATH }[mode])
        df['image'] = df['image'].apply(lambda x: os.path.join(self.dir_dict['data_folder'], x))
        if mode not in ('isic16_test', 'isic20_test'):
            if self.task == 'ben_mal':
                df.replace(to_replace=BEN_MAL_MAP, inplace=True)
            elif self.task in ('nev_mel', '5cls'):
                if self.task == 'nev_mel':
                    df.drop(df[df['class'].isin(['NNV', 'SUS', 'NMC'])].index, errors='ignore', inplace=True)
                df.drop(df[df['class'] == 'UNK'].index, errors='ignore', inplace=True)
            if self.image_type != 'both':  # Keep derm or clinic, samples.
                df.drop(df[df['image_type'] != self.image_type].index, errors='ignore', inplace=True)
        return df

    def log_freqs_per_class(self):
        os.makedirs(os.path.join(MAIN_DIR, 'data_info'), exist_ok=True)
        for key in ('train', 'validation', 'test'):
            df = self.data_df[key]
            with pd.ExcelWriter(os.path.join(MAIN_DIR, 'data_info', 'descr_{}_{}_{}.xlsx'.format(self.task, self.image_type, key)), mode='w') as writer:
                for feature in ['sex', 'age_approx', 'location']:
                    logs_df = df[['image_type', 'class', feature]].value_counts(sort=False, dropna=False).to_frame('counts')
                    logs_df = logs_df.pivot_table(values='counts', fill_value=0, index=['image_type', feature], columns='class', aggfunc=sum)
                    logs_df = logs_df[self.class_names]
                    logs_df.to_excel(writer, sheet_name=feature)

    def oversampling(self):
        max_size = max(self.data_df['train'][['image_type', 'class']].value_counts(sort=False, dropna=False))
        list_of_sub_df = []
        for _image_type in self.data_df['train']['image_type'].unique():
            for _class in self.data_df['train']['class'].unique():
                sub_df = self.data_df['train'].loc[(self.data_df['train']['image_type'] == _image_type) & (self.data_df['train']['class'] == _class)]
                sub_df = pd.concat([sub_df] * (max_size // len(sub_df)))
                sub_df = pd.concat([sub_df, sub_df.sample(n=max_size - len(sub_df))])
                list_of_sub_df.append(sub_df)
        self.data_df['train'] = pd.concat(list_of_sub_df)
        return self.data_df['train']

    def make_onehot(self, df, mode, no_image_type, only_image):
        ohe_features = {'image_path': df['image']}
        if not only_image:
            categories = [LOCATIONS, SEX, AGE_APPROX]
            columns = ['location', 'sex', 'age_approx']
            if not no_image_type:
                categories.append(IMAGE_TYPE)
                columns.append('image_type')
            ohe = OneHotEncoder(handle_unknown='ignore', categories=categories).fit(self.data_df['train'][columns])
            ohe_features['clinical_data'] = ohe.transform(df[columns]).toarray()
        if mode in ('isic16_test', 'isic20_test'):
            # labels = ohe_features['image_path']
            labels = None
        else:
            label_enc = OneHotEncoder(categories=[self.class_names])
            label_enc.fit(self.data_df['train']['class'].values.reshape(-1, 1))
            labels = {'class': label_enc.transform(df['class'].values.reshape(-1, 1)).toarray()}
        return ohe_features, labels
        # return ohe_features, labels

    def get_dataset(self, mode=None, batch=16, no_image_type=False, only_image=False):
        data = self.data_df[mode]
        if mode == 'train':
            data = self.oversampling()
        data = self.make_onehot(df=data, mode=mode, no_image_type=no_image_type, only_image=only_image)
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if mode == 'train':  # Data batches
            dataset = dataset.shuffle(buffer_size=len(self.data_df['train']), reshuffle_each_iteration=True)
        dataset = dataset.batch(batch)
        dataset = dataset.map(lambda sample, label: (self.read_image(sample=sample), label), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if mode == 'train':
            dataset = dataset.map(lambda sample, label: (self.augm(sample), label), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if mode in ('isic16_test', 'isic20_test'):
            dataset = dataset.map(lambda sample, label: sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        if mode == 'train':
            dataset = dataset.repeat()
        else:
            dataset = dataset.repeat(1)
        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def all_datasets(self, batch=16, no_image_type=False, only_image=False):
        return {'train': self.get_dataset('train', batch=batch, no_image_type=no_image_type, only_image=only_image),
                'validation': self.get_dataset('validation', batch=batch, no_image_type=no_image_type, only_image=only_image),
                'test': self.get_dataset('test', batch=batch, no_image_type=no_image_type, only_image=only_image),
                'isic20_test': self.get_dataset('isic20_test', batch=batch, no_image_type=no_image_type, only_image=only_image)}

    def read_image(self, sample):
        def _read_image(x):
            return tf.cast(tf.image.decode_image(tf.io.read_file(x), channels=3, dtype=tf.uint8), dtype=tf.float32)
        sample['image'] = tf.map_fn(fn=lambda x: tf.reshape(tensor=_read_image(x), shape=self.input_shape),
                                    elems=sample['image_path'], fn_output_signature=tf.float32)
        return sample

    def augm(self, sample):
        def _augm(image):
            image = tf.image.random_flip_up_down(image=image)
            image = tf.image.random_flip_left_right(image=image)
            image = tf.image.random_brightness(image=image, max_delta=60.)
            image = tf.image.random_contrast(image=image, lower=.5, upper=1.5)
            image = tf.clip_by_value(image, clip_value_min=0., clip_value_max=255.)
            image = tf.image.random_saturation(image=image, lower=0.8, upper=1.2)
            image = tfa.image.sharpness(image=tf.cast(image, dtype=tf.float32), factor=self.TF_RNG.uniform(shape=[1], minval=0.5, maxval=1.5), name='Sharpness')
            trans_val = self.input_shape[0] * 0.2
            image = tfa.image.translate(images=image, translations=self.TF_RNG.uniform(shape=[2], minval=-trans_val, maxval=trans_val,
                                                                                       dtype=tf.float32), name='Translation')
            image = tfa.image.rotate(images=image, angles=tf.cast(self.TF_RNG.uniform(shape=[], minval=0, maxval=360,
                                                                                      dtype=tf.int32), dtype=tf.float32), interpolation='bilinear', name='Rotation')
            image = tf.cond(tf.less(self.TF_RNG.uniform(shape=[1]), 0.5),
                            lambda: tfa.image.gaussian_filter2d(image=image, sigma=1.5, filter_shape=3, name='Gaussian_filter'),
                            lambda: image)
            return image

        sample['image'] = tf.map_fn(fn=_augm, elems=sample['image'], fn_output_signature=tf.float32)
        cutout_ratio = 0.15
        for i in range(3):
            mask_height = tf.cast(self.TF_RNG.uniform(shape=[], minval=0, maxval=self.input_shape[0] * cutout_ratio),
                                  dtype=tf.int32) * 2
            mask_width = tf.cast(self.TF_RNG.uniform(shape=[], minval=0, maxval=self.input_shape[0] * cutout_ratio),
                                 dtype=tf.int32) * 2
            sample['image'] = tfa.image.random_cutout(sample['image'], mask_size=(mask_height, mask_width))
        sample['image'] = {'xept': xception.preprocess_input, 'incept': inception_v3.preprocess_input,
                           'effnet0': efficientnet.preprocess_input, 'effnet1': efficientnet.preprocess_input,
                           'effnet6': efficientnet.preprocess_input}[self.pretrained](sample['image'])
        return sample
