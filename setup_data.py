import pandas as pd
import os

def merge_datasets():
    columns = ['dataset_id', 'image', 'image_type', 'sex', 'age_approx', 'anatom_site_general', 'class']
    seven_pt = pd.read_csv('data/7pt.csv')
    dermofit = pd.read_csv('data/dermofit.csv')
    isic19 = pd.read_csv('data/isic19.csv')
    isic20 = pd.read_csv('data/isic20.csv')
    mednode = pd.read_csv('data/mednode.csv')
    ph2 = pd.read_csv('data/ph2.csv')
    up = pd.read_csv('data/up.csv')
    datasets = [seven_pt, dermofit, isic19, isic20, mednode, ph2, up]
    for dataset in datasets:  # Drop columns that are not used for training and append missing columns with None values.
        drop_list = list(set(dataset.keys()) - set(columns))
        dataset.drop(drop_list, axis=1, inplace=True)
        for column in columns:
            if column not in dataset:
                dataset[column] = None
    return pd.concat(datasets, ignore_index=True)


def string_to_num(dataset):  # Change string values to numeric.
    image_type_map = {0: 'clinic', 1: 'derm'}
    dataset['image_type'] = dataset['image_type'].map(image_type_map)
    for idx in dataset.index:
        dataset.loc[idx, 'image'] = os.path.join('data', dataset.loc[idx, 'dataset_id'], 'data',
                                                 dataset.loc[idx, 'image'])
    return dataset


def fix_image_names(dataset):  # Fix isic19 and isic20 file names.
    dataset.loc[dataset['dataset_id'] == 'isic19', 'image'] = dataset.loc[
                                                                  dataset['dataset_id'] == 'isic19', 'image'] + '.jpg'
    dataset.loc[dataset['dataset_id'] == 'isic20', 'image'] = dataset.loc[
                                                                  dataset['dataset_id'] == 'isic20', 'image'] + '.jpg'
    return dataset


def setup_data(to_file):
    fix_image_names(string_to_num(merge_datasets())).to_csv(to_file)


setup_data('all_data.csv')
