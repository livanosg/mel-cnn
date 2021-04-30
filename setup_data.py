import pandas as pd
from config import COLUMNS


def setup_datasets(save=False):
    seven_pt = pd.read_csv('data/7pt.csv')
    dermofit = pd.read_csv('data/dermofit.csv')
    isic19 = pd.read_csv('data/isic19.csv')
    isic20 = pd.read_csv('data/isic20.csv')
    mednode = pd.read_csv('data/mednode.csv')
    ph2 = pd.read_csv('data/ph2.csv')
    up = pd.read_csv('data/up.csv')
    datasets = [seven_pt, dermofit, isic19, isic20, mednode, ph2, up]
    for dataset in datasets:
        for column in COLUMNS:
            if column not in dataset:
                dataset[column] = None
    all_data = pd.concat(datasets, ignore_index=True)
    image_type_map = {0: 'clinic', 1: 'derm'}  # Change string values to numeric.
    all_data['image_type'] = all_data['image_type'].map(image_type_map)
    for dataset_id in all_data['dataset_id'].unique():
        all_data.loc[all_data['dataset_id'] == dataset_id, 'image'] =\
            all_data.loc[all_data['dataset_id'] == dataset_id, 'image'].map(lambda x: f'data/{dataset_id}/data/{x}')
    all_data.loc[:, 'age_approx'].fillna(-10, inplace=True)
    all_data.loc[:, "age_approx"] = round(all_data.loc[:, "age_approx"] / 10) * 10
    if save:
        all_data.to_csv('all_data_init.csv', index=False, columns=COLUMNS)
    return datasets


if __name__ == '__main__':
    setup_datasets(save=True)
    exit()
