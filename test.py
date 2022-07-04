from custom_metrics import calc_metrics
from data_prep import get_dataset


def test_fn(args, dirs, model):

    """ run validation and tests"""
    assert args['image_type'] in ('clinic', 'derm')
    thr_d, thr_f1 = calc_metrics(args=args, dirs=dirs, model=model,
                                 dataset=get_dataset(args=args, dataset='validation', dirs=dirs),
                                 dataset_name='validation')
    test_datasets = {'derm': ['isic16_test', 'isic17_test', 'isic18_val_test',
                              'mclass_derm_test', 'up_test'],
                     'clinic': ['up_test', 'dermofit_test', 'mclass_clinic_test']}
    if args['task'] == 'nev_mel':
        test_datasets['derm'].remove('isic16_test')

    for test_dataset in test_datasets[args['image_type']]:
        calc_metrics(args=args, dirs=dirs, model=model,
                     dataset=get_dataset(args=args, dataset=test_dataset, dirs=dirs),
                     dataset_name=test_dataset, dist_thresh=thr_d, f1_thresh=thr_f1)
    if args['task'] == 'ben_mal':
        calc_metrics(args=args, dirs=dirs, model=model,
                     dataset=get_dataset(args=args, dataset='isic20_test', dirs=dirs),
                     dataset_name='isic20_test', dist_thresh=thr_d, f1_thresh=thr_f1)
