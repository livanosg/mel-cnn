import pandas as pd

file = pd.read_csv('all_data.csv')

# print(file['image_type'].value_counts())
print(file['sex'].value_counts())
