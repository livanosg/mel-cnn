import os.path

import numpy as np
import pandas as pd

isic18 = pd.read_csv("/home/livanosg/projects/mel-cnn/data/isic18/meta/ISIC2018_Task3_Validation_GroundTruth.csv")
isic19 = pd.read_csv("/home/livanosg/projects/mel-cnn/data/isic19/ISIC_2019_Training_Metadata.csv")
isic20 = pd.read_csv("/home/livanosg/projects/mel-cnn/data/isic20/meta/ISIC_2020_Training_GroundTruth.csv")
seven_pt = pd.read_csv("/home/livanosg/projects/mel-cnn/data/7pt/meta/meta.csv")
seven_pt_val_indexes = pd.read_csv("/home/livanosg/projects/mel-cnn/data/7pt/meta/valid_indexes.csv")
seven_pt_test_indexes = pd.read_csv("/home/livanosg/projects/mel-cnn/data/7pt/meta/test_indexes.csv")
dermofit = pd.read_csv("/home/livanosg/projects/mel-cnn/data/dermofit/meta/labels.csv")
mednode = pd.read_csv("/home/livanosg/projects/mel-cnn/data/mednode/meta/labels.csv")
ph2 = pd.read_csv("/home/livanosg/projects/mel-cnn/data/ph2/meta/labels.csv")
up = pd.read_csv("/home/livanosg/projects/mel-cnn/data/up/meta/up_data.csv")

isic20_ids = isic20["patient_id"].unique()
isic20_train_ratio = int(len(isic20_ids) * 0.8)
isic20_train_ids = isic20_ids[:isic20_train_ratio]
isic20_val_ids = isic20_ids[isic20_train_ratio:]
isic20_train = isic20[isic20["patient_id"].isin(isic20_train_ids)]
isic20_val = isic20[isic20["patient_id"].isin(isic20_val_ids)]

nans_isic19 = isic19[isic19["lesion_id"].isna()]
isic19 = isic19[~isic19.index.isin(nans_isic19.index)]
isic19_ids = isic19["lesion_id"].unique()
isic19_train_ratio = int(len(isic19_ids) * 0.85)
isic19_train_ids = isic19_ids[:isic19_train_ratio]
isic19_val_ids = isic19_ids[isic19_train_ratio:]
isic19_train = nans_isic19.append(isic19[isic19["lesion_id"].isin(isic19_train_ids)])
isic19_val = isic19[isic19["lesion_id"].isin(isic19_val_ids)]

isic18_val = isic18

seven_pt_val = seven_pt[seven_pt.index.isin(seven_pt_val_indexes["indexes"])]
seven_pt_test = seven_pt[seven_pt.index.isin(seven_pt_test_indexes["indexes"])]
seven_pt_train = seven_pt[~seven_pt.index.isin(seven_pt_val.append(seven_pt_test).index)]

mednode_train = mednode.sample(frac=0.8)
mednode_val = mednode[~mednode.index.isin(mednode_train.index)].sample(frac=0.5)
mednode_test = mednode[~mednode.index.isin(mednode_train.append(mednode_val).index)]
# print(len(mednode_train))
# print(len(mednode_val))
# print(len(mednode_test))

up_patients = up["patient_id"].unique()
up_train = up[up["patient_id"].isin(up_patients[:int(len(up_patients) * 0.8)])]
up_val = up[~up.index.isin(up_train.index)].sample(frac=0.5)
up_test = up[~up.index.isin(up_train.append(up_val).index)]
