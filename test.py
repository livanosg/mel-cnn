from data_prep import MelData
from custom_metrics import calc_metrics


def test_fn(args, dirs, model):
    """ run validation and tests"""
    if args['image_type'] not in ('clinic', 'derm'):
        raise ValueError(f'{args["image_type"]} not valid. Select on one of ("clinic", "derm")')
    data = MelData(args, dirs)
    data.args['clinic_val'] = False
    thr_d, thr_f1 = calc_metrics(args=args, dirs=dirs, model=model,
                                 dataset=data.get_dataset(dataset_name='validation'),
                                 dataset_name='validation')
    test_datasets = {'derm': ['isic16_test', 'isic17_test', 'isic18_val_test',
                              'mclass_derm_test', 'up_test'],
                     'clinic': ['up_test', 'dermofit_test', 'mclass_clinic_test']}
    if args['task'] == 'nev_mel':
        test_datasets['derm'].remove('isic16_test')

    for test_dataset in test_datasets[args['image_type']]:
        calc_metrics(args=args, dirs=dirs, model=model,
                     dataset=data.get_dataset(dataset_name=test_dataset),
                     dataset_name=test_dataset, dist_thresh=thr_d, f1_thresh=thr_f1)
        if args['task'] == 'ben_mal':
            calc_metrics(args=args, dirs=dirs, model=model,
                         dataset=data.get_dataset(dataset_name='isic20_test'),
                         dataset_name='isic20_test', dist_thresh=thr_d, f1_thresh=thr_f1)
