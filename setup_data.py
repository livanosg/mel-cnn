# import os
import pandas as pd


def setup_datasets(save=False):
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
    datasets = pd.concat(datasets, ignore_index=True)
    image_type_map = {0: 'clinic', 1: 'derm'}  # Change string values to numeric.
    datasets['image_type'] = datasets['image_type'].map(image_type_map)
    for dataset_id in datasets['dataset_id'].unique():
        datasets.loc[datasets['dataset_id'] == dataset_id, 'image'] =\
            datasets.loc[datasets['dataset_id'] == dataset_id, 'image'].map(lambda x: f'data/{dataset_id}/data/{x}')
    if save:
        datasets.to_csv('all_data_init.csv', index=False)
    return datasets


def prep_data(file='all_data_init.csv'):
    all_data = pd.read_csv(file)
    all_data.loc[:, 'age_approx'].fillna(-10, inplace=True)
    all_data = pd.get_dummies(all_data, columns=['image_type', 'sex', 'anatom_site_general', 'class'], dtype=int)
    all_data.to_csv('all_data_v2.csv', index=False)


if __name__ == '__main__':
    setup_datasets(save=True)
    prep_data()
    exit()
