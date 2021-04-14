import os
import sys
from datetime import datetime
from hyperparameters import HWC_DOM, BATCH_SIZE_RANGE, DROPOUT_LST, RELU_A, OPTIMIZER_LST, LR_LST, MODEL_LST
from train_model import training

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
run_num = 0
for LR in LR_LST.domain.values:
    for batch_size in BATCH_SIZE_RANGE.domain.values:
        for optimizer in OPTIMIZER_LST.domain.values:
            for HWC in HWC_DOM.domain.values:
                for activation_a in RELU_A.domain.values:
                    for model in MODEL_LST.domain.values:
                        for dropout_rate in (DROPOUT_LST.domain.min_value, DROPOUT_LST.domain.max_value):
                            hparams = {LR_LST: LR,
                                       BATCH_SIZE_RANGE: batch_size,
                                       HWC_DOM: HWC,
                                       RELU_A: activation_a,
                                       DROPOUT_LST: dropout_rate,
                                       OPTIMIZER_LST: optimizer,
                                       MODEL_LST: model}
                            hparams_dict = {h.name: hparams[h] for h in hparams}
                            print(hparams_dict)
                            log_dir = f'logs/run-{str(run_num).zfill(4)}-{datetime.now().strftime("%d%m%y%H%M%S")}'
                            if not os.path.exists(log_dir):
                                os.makedirs(log_dir)
                            with open(log_dir + '/hyperparams.txt', 'a') as f:
                                print(hparams_dict, file=f)
                            training(nodes=sys.argv[1], hparams=hparams, log_dir=log_dir)
                            run_num += 1
