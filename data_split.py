import os
import numpy as np
import pandas as pd
from config import DATA_DIR, COLUMNS, TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH, ISIC20_TEST_PATH, DATA_MAP, MAIN_DIR

NP_RNG = np.random.default_rng(seed=1312)

isic18_val = pd.read_csv(os.path.join(DATA_DIR, 'isic18.csv'))
[isic18_val.insert(loc=0, column=column, value=None) for column in COLUMNS if column not in isic18_val.columns]
isic18_val = isic18_val[COLUMNS]
isic19 = pd.read_csv(os.path.join(DATA_DIR, 'isic19.csv'))
[isic19.insert(loc=0, column=column, value=None) for column in COLUMNS if column not in isic19.columns]
isic19 = isic19[COLUMNS]
isic20 = pd.read_csv(os.path.join(DATA_DIR, 'isic20.csv'))
[isic20.insert(loc=0, column=column, value=None) for column in COLUMNS if column not in isic20.columns]
isic20 = isic20[COLUMNS]
isic20_duplicates = pd.read_csv(os.path.join(DATA_DIR, 'ISIC_2020_Training_Duplicates.csv'))
isic20_duplicates["image_name_2"] = isic20_duplicates["image_name_2"] + ".jpg"
isic20 = isic20[~isic20["image"].isin(isic20_duplicates["image_name_2"])]
mednode = pd.read_csv(os.path.join(DATA_DIR, 'mednode.csv'))
[mednode.insert(loc=0, column=column, value=None) for column in COLUMNS if column not in mednode.columns]
mednode = mednode[COLUMNS]
spt = pd.read_csv(os.path.join(DATA_DIR, '7pt.csv'))
[spt.insert(loc=0, column=column, value=None) for column in COLUMNS if column not in spt.columns]
spt = spt[COLUMNS]
spt_val_idx = pd.read_csv(os.path.join(DATA_DIR, '7pt', 'meta', 'valid_indexes.csv'))
spt_val_idx.index = spt_val_idx["indexes"]
spt_test_idx = pd.read_csv(os.path.join(DATA_DIR, '7pt', 'meta', 'test_indexes.csv'))
spt_test_idx.index = spt_test_idx["indexes"]
dermofit = pd.read_csv(os.path.join(DATA_DIR, 'dermofit.csv'))
[dermofit.insert(loc=0, column=column, value=None) for column in COLUMNS if column not in dermofit.columns]
dermofit = dermofit[COLUMNS]
ph2 = pd.read_csv(os.path.join(DATA_DIR, 'ph2.csv'))
[ph2.insert(loc=0, column=column, value=None) for column in COLUMNS if column not in ph2.columns]
ph2 = ph2[COLUMNS]
up = pd.read_csv(os.path.join(DATA_DIR, 'up.csv'))
[up.insert(loc=0, column=column, value=None) for column in COLUMNS if column not in up.columns]
up = up[COLUMNS]
padufes = pd.read_csv(os.path.join(DATA_DIR, 'padufes.csv'))


padufes_ids = padufes["patient_id"].unique()
NP_RNG.shuffle(padufes_ids)
padufes_train = padufes.loc[padufes["patient_id"].isin(padufes_ids[:int(len(padufes_ids) * 0.9)])]
padufes_val = padufes.loc[padufes["patient_id"].isin(padufes_ids[int(len(padufes_ids) * 0.9):])]

isic16_test = pd.read_csv(os.path.join(DATA_DIR, 'isic16_test.csv'))
isic20_test = pd.read_csv(os.path.join(DATA_DIR, 'isic20_test.csv'))
nans_isic19 = isic19[isic19["lesion_id"].isna()]
isic19_not_nans = isic19[~isic19.index.isin(nans_isic19.index)]
isic19_ids = isic19_not_nans["lesion_id"].unique()
NP_RNG.shuffle(isic19_ids)
isic19_train = isic19.loc[isic19["lesion_id"].isin(isic19_ids[:int(len(isic19_ids) * 0.9)])].append(nans_isic19)
isic19_val = isic19.loc[isic19["lesion_id"].isin(isic19_ids[int(len(isic19_ids) * 0.9):])]  # int(len(isic19_ids) * 0.9)
isic20_ids = isic20["patient_id"].unique()
NP_RNG.shuffle(isic20_ids)
isic20_train = isic20.loc[isic20["patient_id"].isin(isic20_ids[:int(len(isic20_ids) * 0.9)])]
isic20_val = isic20.loc[isic20["patient_id"].isin(isic20_ids[int(len(isic20_ids) * 0.9):])]
spt_test = spt[spt.index.isin(spt_test_idx.index)]
spt_val = spt[spt.index.isin(spt_val_idx.index)].append(spt_test)
spt_train = spt[~spt.index.isin(spt_val.index)]
mednode_train = mednode.sample(frac=0.9, random_state=NP_RNG.bit_generator)
mednode_val = mednode[~mednode.index.isin(mednode_train.index)]
ph2_train = ph2.sample(frac=0.9, random_state=NP_RNG.bit_generator)
ph2_val = ph2[~ph2.index.isin(ph2_train.index)]

total_train = padufes_train.append(isic19_train).append(isic20_train).append(spt_train).append(mednode_train).append(ph2_train)
total_val = padufes_val.append(isic19_val).append(isic20_val).append(spt_val).append(mednode_val).append(ph2_val)
total_test = isic18_val.append(dermofit).append(up)
total_data_len = len(total_train) + len(total_val) + len(total_test)

for df, save_to in [(total_train, TRAIN_CSV_PATH), (total_val, VAL_CSV_PATH), (total_test, TEST_CSV_PATH), (isic20_test, ISIC20_TEST_PATH)]:
    df = df.sample(frac=1., random_state=NP_RNG.bit_generator)
    columns = ['dataset_id', 'location', 'sex', 'image', 'age_approx', 'image_type', 'class']
    df['age_approx'] -= (df['age_approx'] % 10)
    df['image'] = df[['dataset_id', 'image']].apply(lambda id_img: os.path.join(id_img[0], 'data', id_img[1]), axis=1)
    df.replace(to_replace=DATA_MAP, inplace=True)

    print("{}| Count:{} Ratio:{}".format(os.path.split(save_to)[-1].rjust(15), str(len(df)).rjust(6), str(round(len(df) / total_data_len, 3)).rjust(6)))
    if os.path.basename(save_to).split('.')[0] == 'isic20_test':
        columns.remove('class')
    df.to_csv(save_to, index=False, columns=columns)

    # log datasets description
    dataset_info_dict = {}
    image_type_inv = {}
    if not os.path.basename(save_to).split('.')[0] == 'isic20_test':
        for dataset_id in df['dataset_id'].unique():
            dataset_part = df[df.loc[:, 'dataset_id'] == dataset_id]  # fraction per class
            dataset_img_type_dict = {}
            for image_type in dataset_part['image_type'].unique():
                dataset_image_part = dataset_part[dataset_part.loc[:, 'image_type'] == image_type]
                dataset_image_part = dataset_image_part.drop(dataset_image_part[dataset_image_part['class'] == 5].index, errors='ignore')
                dataset_class_dict = {}
                for k, v in dataset_image_part['class'].value_counts().items():
                    dataset_class_dict[k] = v
                dataset_img_type_dict[image_type] = dataset_class_dict

            dataset_info_dict[dataset_id] = dataset_img_type_dict
        info = pd.DataFrame(dataset_info_dict).stack().apply(pd.Series)
        info = info[sorted(info.columns)]
        info.fillna(0, inplace=True)
        save_path = os.path.join(MAIN_DIR, 'data_info', os.path.basename(save_to).split('.')[0]+'_info')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        info.to_html(save_path + '.html', bold_rows=False, border=4)
        info.to_csv(save_path + '.csv')
