import os

import pandas as pd

main_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(main_dir, "data")

isic18 = pd.read_csv(os.path.join(data_dir, 'isic18.csv'))
isic19 = pd.read_csv(os.path.join(data_dir, 'isic19.csv'))
isic20 = pd.read_csv(os.path.join(data_dir, 'isic20.csv'))
isic20_duplicates = pd.read_csv(os.path.join(data_dir, 'ISIC_2020_Training_Duplicates.csv'))
isic20_duplicates["image_name_2"] = isic20_duplicates["image_name_2"] + ".jpg"
isic20 = isic20[~isic20["image"].isin(isic20_duplicates["image_name_2"])]

mednode = pd.read_csv(os.path.join(data_dir,  'mednode.csv'))
spt = pd.read_csv(os.path.join(data_dir,  '7pt.csv'))
spt_val_idx = pd.read_csv(os.path.join(data_dir, '7pt', 'meta', 'valid_indexes.csv'))
spt_test_idx = pd.read_csv(os.path.join(data_dir, '7pt', 'meta', 'test_indexes.csv'))
dermofit = pd.read_csv(os.path.join(data_dir,  'dermofit.csv'))
ph2 = pd.read_csv(os.path.join(data_dir, 'ph2.csv'))
up = pd.read_csv(os.path.join(data_dir, 'up.csv'))

isic18_val = isic18
nans_isic19 = isic19[isic19["lesion_id"].isna()]
isic19_not_nans = isic19[~isic19.index.isin(nans_isic19.index)]
isic19_ids = sorted(isic19_not_nans["lesion_id"].unique())
isic19_train_ratio = int(len(isic19_ids) * 0.70)
isic19_val_ratio = int(len(isic19_ids) * 0.85)
isic19_train_ids = isic19_ids[:isic19_train_ratio]
isic19_val_ids = isic19_ids[isic19_train_ratio:isic19_val_ratio]
isic19_test_ids = isic19_ids[isic19_val_ratio:]
isic19_train = nans_isic19.append(isic19[isic19["lesion_id"].isin(isic19_train_ids)])
isic19_val = isic19[isic19["lesion_id"].isin(isic19_val_ids)]
isic19_test = isic19[isic19["lesion_id"].isin(isic19_test_ids)]

isic20_ids = sorted(isic20["patient_id"].unique())
isic20_train_len = int(len(isic20_ids) * 0.8)
isic20_val_len = int(len(isic20_ids) * 0.9)
isic20_train_ids = isic20_ids[:isic20_train_len]
isic20_val_ids = isic20_ids[isic20_train_len:isic20_val_len]
isic20_test_ids = isic20_ids[isic20_val_len:]
isic20_train = isic20[isic20["patient_id"].isin(isic20_train_ids)]
isic20_val = isic20[isic20["patient_id"].isin(isic20_val_ids)]
isic20_test = isic20[isic20["patient_id"].isin(isic20_test_ids)]

spt_val = spt[spt.index.isin(spt_val_idx["indexes"])]
spt_test = spt[spt.index.isin(spt_test_idx["indexes"])]
spt_train = spt[~spt.index.isin(spt_val_idx["indexes"].append(spt_test_idx["indexes"]))]

dermofit_val = dermofit.sample(frac=.5)
dermofit_test = dermofit[~dermofit.index.isin(dermofit_val.index)]

mednode_train = mednode.sample(frac=0.8)
mednode_val = mednode[~mednode.index.isin(mednode_train.index)].sample(frac=0.5)
mednode_test = mednode[~mednode.index.isin(mednode_train.append(mednode_val).index)]

ph2_train = ph2.sample(frac=.8)
ph2_val = ph2[~ph2.index.isin(ph2_train.index)].sample(frac=.5)
ph2_test = ph2[~ph2.index.isin(ph2_train.append(ph2_val).index)]

# up_patients = up["patient_id"].unique()
up_train = up  # [up["patient_id"].isin(up_patients[:int(len(up_patients) * 0.8)])]
# up_val = up[up["patient_id"].isin(up_patients[int(len(up_patients) * 0.8):int(len(up_patients) * 0.9)])]
# up_test = up[up["patient_id"].isin(up_patients[int(len(up_patients) * 0.9):])]

total_train = isic19_train.append(isic20_train).append(spt_train).append(mednode_train).append(ph2_train).append(up_train)
total_val = isic18_val.append(isic19_val).append(isic20_val).append(mednode_val).append(ph2_val).append(spt_val).append(dermofit_val)  # .append(up_val)
total_test = isic19_test.append(isic20_test).append(mednode_test).append(ph2_test).append(spt_test).append(dermofit_test)  # .append(up_test)
total_data_len = len(total_train) + len(total_val) + len(total_test)
total_train.to_csv("train.csv", index=False, columns=["dataset_id", "class", "anatom_site_general", "sex", "image", "age_approx", "image_type"])
total_val.to_csv("val.csv", index=False, columns=["dataset_id", "class", "anatom_site_general", "sex", "image", "age_approx", "image_type"])
total_test.to_csv("test.csv", index=False, columns=["dataset_id", "class", "anatom_site_general", "sex", "image", "age_approx", "image_type"])
